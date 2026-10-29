#!/usr/bin/env python3
import asyncio
import os
import sys
import json
import logging
import math
import re
import time
import requests
from collections import defaultdict
from typing import Dict, Any, List

from botcore import _parse_bool_flag, load_env_stack, safe_nonneg_float
from cachepaths import normalize_venue_name
from notify_engine.alertnotifiers import (
    _alert_attr,
    _external_notifications_disabled,
    _safe_float,
    is_fake_alert_text,
)
from notify_engine.mailer import (
    is_configured as email_is_configured,
    is_email_mirrored,
    last_error_is_auth_failure,
    send_email,
)

_MAX_QUEUE_RETRIES = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] Orchestrator: %(message)s")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_GUARD_MARKERS = (
    "🛑", "🛡", "STOP-LOSS", "STOP_LOSS", "TRAILING", "LIQUID", "CATASTROPH", "CRASH",
    "LICHID", "CATASTROF", "IMBALANCE", "DEZECHILIBR", "QUARANTINE", "SL ", " SL",
)
_OPS_MARKERS = (
    "FAILED", "ERROR", "MANUAL", "GONE",
    "ESUAT", "ERORI", "DISPARUT",
)
_TRADE_MARKERS = (
    "BUY", "SELL", "FILLED", "TREND_ENTRY", "ADOPT", "EXECUTION", "📝", "🎉",
)
_EXPLICIT_ERROR_MARKERS = (
    "FAILED", "ERROR", "EXCEPTION", "CRASH", "ESUAT", "ERORI", "DISPARUT", "GONE",
)
_SERVER_MARKERS = (
    "CONFIG RELOAD", "SERVER STATUS", "SERVER DOWN", "DEADMAN",
)

def _ensure_env_loaded() -> None:
    env_p = os.path.join(ROOT_DIR, ".env")
    if os.path.isfile(env_p):
        load_env_stack(env_p)
    else:
        cfg_p = os.path.join(ROOT_DIR, "config.env")
        if os.path.isfile(cfg_p):
            load_env_stack(cfg_p)


def resolve_topic(category: str) -> str:
    from botcore import required_env
    cat = ("" if isinstance(category, bool) else str(category or "")).strip().upper()
    if cat in ("SERVER", "SERVER_STATE", "SERVER_STATUS", "DEADMAN"):
        cat = "SERVER"
    key = f"NTFY_TOPIC_{cat}"
    if not os.environ.get(key, "").strip():
        _ensure_env_loaded()
    return required_env(key)

def _category_for_title_and_source(title: str, source: str) -> str:
    t = ("" if isinstance(title, bool) else str(title or "")).upper()
    s = ("" if isinstance(source, bool) else str(source or "")).lower()
    if "AUTO_TECHNICAL_REPAIR" in t or s == "ai_supervisor":
        if any(m in t for m in _EXPLICIT_ERROR_MARKERS):
            return "ERROR"
        return "SERVER"
    if "MACRO" in t or "macro" in s or "geopolitical" in s:
        return "MACRO"
    if any(m in t for m in _GUARD_MARKERS) or any(m in s for m in ("guard", "trail", "assetguardian", "stop")):
        return "GUARD"
    if any(m in t for m in _SERVER_MARKERS) or "deadman" in s:
        return "SERVER"

    # Real trades (order launched, order executed, fills) route to TRADES unless explicitly failed/errored
    has_trade_marker = any(m in t for m in _TRADE_MARKERS) or s in ("tradeall", "rtrade", "monitororder")
    has_explicit_error = any(m in t for m in _EXPLICIT_ERROR_MARKERS) or "watchdog" in s
    if has_trade_marker and not has_explicit_error:
        return "TRADES"

    # Strip owner tags like [manual] or (manual) so operational error check is not tricked by order attribution
    t_no_owner = re.sub(
        r"\[(MANUAL|TRADEALL|MONITORORDER|RTRADE|MONITORTRADES|ASSETGUARDIAN)\]|\((MANUAL|TRADEALL|MONITORORDER|RTRADE|MONITORTRADES|ASSETGUARDIAN)\)",
        "",
        t,
        flags=re.IGNORECASE,
    )
    if any(m in t_no_owner for m in _OPS_MARKERS) or "watchdog" in s:
        return "ERROR"

    if (
        "alert" in s or "price" in s or "prag" in t.lower() or "threshold" in t.lower()
        or "coin" in t.lower() or any(src in s for src in ("coinmarketcap", "coingecko", "dexscreener", "pricechecker", "notifier"))
        or any(m in t for m in ("▲", "▼", "RISE", "DROP"))
    ):
        return "PRICE"
    return "TRADES"

def _topic_for_category(title: str, source: str) -> str:
    cat = _category_for_title_and_source(title, source)
    return resolve_topic(cat)

def _resolve_provider_label(bot_name: str = "", source: str = "") -> str:
    bot_str = "" if isinstance(bot_name, bool) else str(bot_name or "")
    src_str = "" if isinstance(source, bool) else str(source or "")
    for cand in (normalize_venue_name(src_str), normalize_venue_name(bot_str)):
        if cand == "kraken":
            return "Kraken"
        if cand == "hyperliquid":
            return "Hyperliquid"
        if cand == "t212":
            return "T212"
        if cand == "binance":
            return "Binance"
    s = src_str.lower()
    b = bot_str.lower()
    if "kraken" in s or "kraken" in b or "xstock" in b:
        return "Kraken"
    if "hyperliquid" in s or "hl" in s or "hyperliquid" in b or b.startswith("hl"):
        return "Hyperliquid"
    if "t212" in s or "trading212" in s or "t212" in b:
        return "T212"
    if "binance" in s or "binance" in b or b in ("rtrade", "tradeall", "monitortrades", "order_retry", "monitororder") or "monitororder" in s:
        return "Binance"
    if "-" in bot_str:
        return bot_str.split("-")[0]
    return bot_str or (src_str.capitalize() if src_str else "")

class NotificationServer:
    def __init__(self):
        self._ensure_environment()
        self.ntfy_token = self._load_ntfy_token()
        self.rules = self._load_rules()
        self.cooldowns: Dict[str, float] = {}
        self.traceback_buffers: Dict[str, List[str]] = defaultdict(list)
        
        # Persistent Queue File
        from cachepaths import cache_path
        self.queue_file = os.environ.get("NOTIFICATION_QUEUE_FILE") or cache_path("notification_queue.jsonl")
        os.makedirs(os.path.dirname(self.queue_file), exist_ok=True)

    def _ensure_environment(self) -> None:
        _ensure_env_loaded()

    def _load_ntfy_token(self) -> str:
        tok = os.environ.get("NTFY_TOKEN", "").strip()
        if not tok:
            _ensure_env_loaded()
            tok = os.environ.get("NTFY_TOKEN", "").strip()
        return tok

    def _load_rules(self) -> List[Dict[str, Any]]:
        rules_path = os.path.join(os.path.dirname(__file__), "rules.json")
        try:
            with open(rules_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for rule in data.get("rules", []):
                    rule["regex_obj"] = re.compile(rule["match_regex"])
                return data.get("rules", [])
        except Exception as e:
            logging.error(f"Failed to load rules.json: {e}")
            return []

    def _resolve_topic(self, category: str) -> str:
        return resolve_topic(category)
        
    def _enqueue_payload(self, intent_type: str, data: dict):
        try:
            payload = dict(data) if isinstance(data, dict) else {}
            payload["__orchestrator_intent__"] = intent_type
            with open(self.queue_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception as e:
            logging.error(f"Failed to enqueue notification: {e}")

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
        from notify_engine.alertnotifiers import pass_common_gate
        allowed, reason, _, out_payload = pass_common_gate(
            "ntfy",
            title,
            body,
            priority,
            topic,
            alerts,
            urgent=urgent,
        )
        if not allowed:
            logging.info(f"ntfy delivery gated by common policy ({reason}): {title}")
            if (
                reason not in ("duplicate", "similar_eliminated", "buffered_for_bulk")
                and is_email_mirrored(topic)
            ):
                self._send_email(title, body)
            return False
        out_title, out_body, out_prio, out_topic = out_payload or (title, body, priority, topic)
        prev = getattr(self, "_in_gated_dispatch", False)
        self._in_gated_dispatch = True
        try:
            return self._send_ntfy(out_title, out_body, out_prio, out_topic)
        finally:
            self._in_gated_dispatch = prev

    def flush_pending_bulk(self, *, now: float | None = None, force: bool = False) -> bool:
        """Flush and synthesize any buffered notifications once the 5-minute common gate opens."""
        from notify_engine.alertnotifiers import pop_due_bulk_batch
        flushed = False
        batch = pop_due_bulk_batch("ntfy", now=now, force=force)
        if batch:
            bulk_title, bulk_body, bulk_prio, bulk_topic = batch
            if not bulk_topic:
                try:
                    bulk_topic = self._resolve_topic("ERROR")
                except Exception:
                    bulk_topic = "ntfy-error-1978"
            prev = getattr(self, "_in_gated_dispatch", False)
            self._in_gated_dispatch = True
            try:
                if self._send_ntfy(bulk_title, bulk_body, bulk_prio, bulk_topic):
                    flushed = True
            finally:
                self._in_gated_dispatch = prev
        email_batch = pop_due_bulk_batch("email", now=now, force=force)
        if email_batch:
            email_subj, email_body, _, _ = email_batch
            if self._send_email(email_subj, email_body, is_retry=False, dedup=False):
                flushed = True
        return flushed

    def _send_ntfy(self, title: str, message: str, priority: str, topic: str, is_retry: bool = False, skip_email: bool = False) -> bool:
        topic_str = "" if isinstance(topic, bool) or topic is None else str(topic).strip()
        if not topic_str:
            return True
        title = "Alert" if isinstance(title, bool) or title is None else (str(title).strip() or "Alert")
        message = "" if isinstance(message, bool) or message is None else str(message)
        priority = "default" if isinstance(priority, bool) or priority is None else (str(priority).strip() or "default")
        if _external_notifications_disabled():
            logging.debug(f"External notifications disabled: would send to {topic_str}: {title}")
            return True

        if is_fake_alert_text(title, message, topic_str):
            logging.info(f"Skipping test/fake alert in _send_ntfy: {title}")
            return True

        if not is_retry and not getattr(self, "_in_gated_dispatch", False):
            from notify_engine.alertnotifiers import pass_common_gate
            allowed, reason, _, out_payload = pass_common_gate(
                "ntfy",
                title,
                message,
                priority,
                topic_str,
                urgent=(priority == "urgent"),
            )
            if not allowed:
                logging.info(f"Direct ntfy gated by common policy ({reason}): {title}")
                if (
                    reason not in ("duplicate", "similar_eliminated", "buffered_for_bulk")
                    and not skip_email
                    and is_email_mirrored(topic_str)
                ):
                    self._send_email(title, message)
                return False
            if out_payload:
                title, message, priority, topic_str = out_payload

        # ERROR / DEADMAN topics are mirrored to email (policy: notify_engine.mailer).
        # Sent before the push so an ntfy outage or quota never swallows the email.
        if not is_retry and not skip_email and is_email_mirrored(topic_str):
            self._send_email(title, message)
        
        topic_str = "" if isinstance(topic_str, bool) else str(topic_str or "").strip()
        if not topic_str:
            return True
        if topic_str.startswith(("http://", "https://")):
            url = topic_str
        else:
            url = f"https://ntfy.sh/{topic_str.lstrip('/')}"
        clean_header_title = re.sub(r"[\r\n]+", " ", "" if isinstance(title, bool) else str(title or "Alert")).strip()
        clean_priority = re.sub(r"[\r\n]+", " ", "" if isinstance(priority, bool) else str(priority or "default")).strip() or "default"
        headers = {
            "Title": clean_header_title.encode("utf-8").decode("latin-1", "ignore"),
            "Priority": clean_priority
        }
        if self.ntfy_token:
            headers["Authorization"] = f"Bearer {self.ntfy_token}"
            
        try:
            resp = requests.post(url, data=message.encode("utf-8"), headers=headers, timeout=5)
            if getattr(resp, "status_code", None) == 429:
                text = getattr(resp, "text", "") or ""
                if "daily" in text.lower() or "42908" in text:
                    logging.warning(f"ntfy daily limit reached: {text}")
                    from notify_engine.alertnotifiers import _mark_provider_daily_limit
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
            logging.info(f"Dispatched ntfy alert to [{topic}] (priority={priority}): {title}")
            return True
        except Exception as e:
            if not is_retry:
                logging.error(f"Failed to send ntfy alert (queueing): {e}")
                self._enqueue_payload("ntfy_webhook", {
                    "title": title,
                    "message": message,
                    "priority": priority,
                    "topic": topic
                })
            return False
            
    def _send_email(self, subject: str, message: str, is_retry: bool = False, *, dedup: bool = True) -> bool:
        # Retries and synthesized bulk batches bypass dedup: the first attempt already reserved this fingerprint.
        ok = send_email(subject, message, dedup=(dedup and not is_retry))
        if not ok and not is_retry and email_is_configured() and not last_error_is_auth_failure():
            self._enqueue_payload("email", {"subject": subject, "message": message})
        return ok

    @staticmethod
    def _format_price_alerts_batch_title(alerts: list) -> str:
        count = len(alerts)
        items = []
        symbols = []
        for a in alerts:
            sym = _alert_attr(a, "symbol", "N/A")
            atype = _alert_attr(a, "alert_type", "")
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
        elif syms_str and len(syms_str) <= 50:
            return f"Price Alerts ({count}): {syms_str}"
        else:
            return f"Price Alerts ({count} coins)"

    @staticmethod
    def _format_new_coins_batch_title(alerts: list) -> str:
        count = len(alerts)
        symbols = []
        for a in alerts:
            sym = _alert_attr(a, "symbol", "N/A")
            if sym and sym != "N/A" and sym not in symbols:
                symbols.append(sym)
        syms_str = ", ".join(symbols)
        if syms_str and len(syms_str) <= 50:
            return f"New Coins ({count}): {syms_str}"
        else:
            return f"New Coins Discovered ({count})"

    def dispatch_alerts(self, alerts: list, webhook_url: str = None, bot_name: str = "") -> bool:
        if not isinstance(alerts, (list, tuple)) or not alerts:
            return False
        bot_name = "" if isinstance(bot_name, bool) else str(bot_name or "")

        first = alerts[0]
        from notify_engine.alertnotifiers import AlertNotifier
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
                    sym = _alert_attr(first, "symbol", "N/A")
                    atype = _alert_attr(first, "alert_type", "alert")
                    pchg = _safe_float(_alert_attr(first, "percent_change", 0.0), 0.0)
                    dir_str = "▲" if atype == "up" else "▼"
                    title = f"{sym} {dir_str} {pchg:+.2f}%"
                else:
                    title = self._format_price_alerts_batch_title(alerts)
                body = AlertNotifier.format_batch_message(alerts)
                source = _alert_attr(first, "source") or "price_alert"
            elif alert_type == "new_coin_discovered":
                if len(alerts) == 1:
                    sym = _alert_attr(first, "symbol", "N/A")
                    title = f"New Coin: {sym}"
                else:
                    title = self._format_new_coins_batch_title(alerts)
                body = AlertNotifier.format_batch_message(alerts)
                source = _alert_attr(first, "source") or "price_alert"
            elif alert_type == "bot_event":
                base_name = _alert_attr(first, "name") or "Alert"
                if len(alerts) == 1:
                    title = base_name
                    body = _alert_attr(first, "body") or ""
                else:
                    title = f"{base_name} ({len(alerts)})"
                    body = AlertNotifier.format_batch_message(alerts)
                source = _alert_attr(first, "source") or ""
            else:
                base_name = _alert_attr(first, "name") or _alert_attr(first, "title") or "Alert"
                if len(alerts) == 1:
                    title = base_name
                    body = _alert_attr(first, "body") or _alert_attr(first, "message") or _alert_attr(first, "symbol") or ""
                else:
                    title = f"{base_name} ({len(alerts)})"
                    body = AlertNotifier.format_batch_message(alerts)
                source = _alert_attr(first, "source") or ""

        title = "" if isinstance(title, bool) else str(title or "Alert")
        body = "" if isinstance(body, bool) else str(body or "")
        source = "" if isinstance(source, bool) else str(source or "")

        topic = None
        if isinstance(webhook_url, str) and webhook_url.strip():
            topic = webhook_url.strip().rstrip("/").rsplit("/", 1)[-1]
        if not topic:
            topic = _topic_for_category(title, source)

        cat = _category_for_title_and_source(title, source)
        price_topic = self._resolve_topic("PRICE")
        trades_topic = self._resolve_topic("TRADES")

        # Strip any existing [price_notifier] or [macro_analyzer] prefix
        cleaned_title = re.sub(
            r"^\[(?:price_notifier|macro_analyzer)(?:\.py)?\]\s*",
            "",
            title,
            flags=re.IGNORECASE,
        ).strip()

        has_auto_topic_tag = bool(re.search(r"\[auto_[a-z0-9_]+", cleaned_title))
        is_macro_bot = bot_name.lower() in ("macro_analyzer", "macro_analyzer.py")
        is_price = (
            topic == price_topic
            or cat == "PRICE"
            or bot_name.lower() in ("price_notifier", "price_notifier.py", "price_alert")
            or str(source).lower() in ("price_alert", "coinmarketcap", "coingecko", "dexscreener", "pricechecker", "notifier")
            or any(m in cleaned_title for m in ("▲", "▼"))
        )

        if is_price or is_macro_bot or has_auto_topic_tag:
            full_title = cleaned_title
        elif topic == trades_topic or cat == "TRADES":
            # For trade notifications, show only provider in brackets without coin
            provider = _resolve_provider_label(bot_name, source)
            if provider:
                if bot_name and cleaned_title.startswith(f"[{bot_name}]"):
                    cleaned_title = cleaned_title[len(f"[{bot_name}]"):].strip()
                if cleaned_title.startswith(f"[{provider}]"):
                    full_title = cleaned_title
                else:
                    full_title = f"[{provider}] {cleaned_title}"
            else:
                full_title = cleaned_title
        else:
            if bot_name and not cleaned_title.startswith(f"[{bot_name}]"):
                full_title = f"[{bot_name}] {cleaned_title}"
            else:
                full_title = cleaned_title

        # Block synthetic/fake test instruments from ever leaking to external topics
        all_syms = " ".join(str(_alert_attr(a, "symbol", "")) for a in alerts)
        if is_fake_alert_text(full_title, body, all_syms):
            logging.info(f"Skipping test/fake alert: {full_title}")
            return True

        from notify_engine.alertnotifiers import _alerts_are_urgent
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

    def process_line(self, line: str, bot_name: str):
        if not isinstance(line, str) or isinstance(line, bool):
            return
        bot_name = "" if isinstance(bot_name, bool) or bot_name is None else str(bot_name)
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
                        lines = self.traceback_buffers[bot_name]
                        accumulated = lines[0] + "\n  ...\n" + "\n".join(lines[-10:])
                        self.traceback_buffers[bot_name] = []
                        line = accumulated
                    else:
                        return
                else:
                    # Terminal exception line reached!
                    self.traceback_buffers[bot_name].append(clean_stripped)
                    lines = self.traceback_buffers[bot_name]
                    if len(lines) > 12:
                        accumulated = lines[0] + "\n  ...\n" + "\n".join(lines[-10:])
                    else:
                        accumulated = "\n".join(lines)
                    self.traceback_buffers[bot_name] = []
                    line = accumulated
        elif "\n" not in clean_stripped and "Traceback (most recent call last):" in clean_stripped:
            self.traceback_buffers[bot_name] = [clean_stripped]
            return

        # 1. Check for Explicit AlertNotifier Intent (JSON)
        if "__orchestrator_intent__" in line:
            try:
                start_idx = line.find("{")
                end_idx = line.rfind("}")
                if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
                    json_str = line[start_idx:end_idx + 1]
                    payload = json.loads(json_str)
                else:
                    payload = json.loads(line)

                intent = payload.get("__orchestrator_intent__") if isinstance(payload, dict) else None
                if intent == "ntfy_webhook":
                    # Direct retry or pre-formatted [auto_...] topic payload
                    if "title" in payload:
                        raw_t = payload.get("title")
                        raw_m = payload.get("message")
                        raw_p = payload.get("priority")
                        raw_top = payload.get("topic")
                        t_val = "Alert" if isinstance(raw_t, bool) or raw_t is None else (str(raw_t).strip() or "Alert")
                        m_val = "" if isinstance(raw_m, bool) or raw_m is None else str(raw_m)
                        p_val = "default" if isinstance(raw_p, bool) or raw_p is None else (str(raw_p).strip() or "default")
                        top_val = (
                            self._resolve_topic("ERROR")
                            if isinstance(raw_top, bool) or raw_top is None or not str(raw_top).strip()
                            else str(raw_top).strip()
                        )
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
                    raw_s = payload.get("subject")
                    raw_em = payload.get("message")
                    s_val = "Alert" if isinstance(raw_s, bool) or raw_s is None else (str(raw_s).strip() or "Alert")
                    em_val = "" if isinstance(raw_em, bool) or raw_em is None else str(raw_em)
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
                if bot_name.lower() in ("price_notifier", "price_notifier.py"):
                    title = "Price Alert"
                else:
                    title = f"[{bot_name}] Alert" if bot_name else "Alert"
                ntfy_alert = {
                    "type": "bot_event",
                    "name": title,
                    "body": msg,
                    "source": bot_name or "orchestrator",
                    "symbol": "SYSTEM",
                }
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
        if is_fake_alert_text(line):
            return
        if any(
            internal in line
            for internal in (
                "Failed to send ntfy alert",
                "Email delivery failed",
                "[Notifier] File exception",
                "Error flushing notification queue",
                "ntfy daily limit reached",
            )
        ):
            return
        for rule in self.rules:
            if rule["regex_obj"].search(line):
                # Enforce Cooldown
                rule_key = f"{bot_name}_{rule['name']}"
                now = time.time()
                last_time = safe_nonneg_float(self.cooldowns.get(rule_key, 0), 0.0)
                if last_time > now:
                    last_time = 0.0
                cd_min = safe_nonneg_float(rule.get("cooldown_minutes", 0), 0.0)
                cooldown_sec = cd_min * 60.0
                
                if now - last_time >= cooldown_sec:
                    topic_cat = rule.get("topic_category", "ERROR").upper()
                    topic = self._resolve_topic(topic_cat)
                    if topic_cat == "PRICE" or bot_name.lower() in ("price_notifier", "price_notifier.py"):
                        full_title = rule["name"]
                    elif topic_cat == "TRADES":
                        provider = _resolve_provider_label(bot_name)
                        full_title = f"[{provider}] {rule['name']}" if provider else rule["name"]
                    else:
                        full_title = f"[{bot_name}] {rule['name']}" if bot_name else rule["name"]

                    rule_alert = {
                        "type": "bot_event",
                        "name": full_title,
                        "body": line.strip(),
                        "source": bot_name or "orchestrator",
                        "symbol": "SYSTEM",
                    }
                    self.cooldowns[rule_key] = now
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
                with open(self.queue_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                
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
                                raw_title = payload.get("title")
                                title = "" if isinstance(raw_title, bool) or raw_title is None else str(raw_title)
                                raw_msg = payload.get("message")
                                msg = "" if isinstance(raw_msg, bool) or raw_msg is None else str(raw_msg)
                                raw_p = payload.get("priority")
                                prio = "default" if isinstance(raw_p, bool) or raw_p is None else (str(raw_p).strip() or "default")
                                raw_top = payload.get("topic")
                                if isinstance(raw_top, bool) or raw_top is None or not str(raw_top).strip():
                                    try:
                                        top = self._resolve_topic("ERROR")
                                    except Exception:
                                        top = "ntfy-error-1978"
                                else:
                                    top = str(raw_top).strip()
                                if not title.startswith("[DELAYED]"):
                                    title = "[DELAYED] " + title
                                success = await asyncio.to_thread(
                                    self._send_ntfy,
                                    title,
                                    msg,
                                    prio,
                                    top,
                                    True
                                )
                            elif "alerts" in payload:
                                success = await asyncio.to_thread(
                                    self.dispatch_alerts,
                                    payload.get("alerts") or [],
                                    payload.get("webhook_url"),
                                    ""
                                )
                            else:
                                success = True
                        elif intent == "email":
                            raw_subj = payload.get("subject")
                            subj = "" if isinstance(raw_subj, bool) or raw_subj is None else str(raw_subj)
                            raw_emsg = payload.get("message")
                            emsg = "" if isinstance(raw_emsg, bool) or raw_emsg is None else str(raw_emsg)
                            if not subj.startswith("[DELAYED]"):
                                subj = "[DELAYED] " + subj
                            success = await asyncio.to_thread(
                                self._send_email,
                                subj,
                                emsg,
                                True
                            )
                        else:
                            success = True  # Unknown intent, drop it
                            
                        if success:
                            success_count += 1
                            queue_dirty = True
                        elif intent == "email" and last_error_is_auth_failure():
                            failed_intents.add(intent)
                            queue_dirty = True
                            logging.warning(
                                f"Dropping queued email due to permanent SMTP authentication failure: {payload.get('subject', '')!r}"
                            )
                        else:
                            from notify_engine.alertnotifiers import _safe_float
                            failed_intents.add(intent)
                            queue_dirty = True
                            retries = max(0, int(_safe_float(payload.get("retry_count", 0), 0.0))) + 1
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
                        try:
                            with open(self.queue_file, "r", encoding="utf-8") as f_cur:
                                cur_lines = f_cur.readlines()
                            if len(cur_lines) > read_count:
                                newly_appended = [
                                    l.strip() for l in cur_lines[read_count:] if l.strip()
                                ]
                        except Exception:
                            pass
                    from state_io import atomic_write_text
                    content = "".join(r + "\n" for r in remaining + newly_appended)
                    atomic_write_text(self.queue_file, content)
                    if success_count > 0:
                        logging.info(f"Flushed {success_count} delayed notifications from queue.")

            except Exception as e:
                logging.error(f"Error flushing notification queue: {e}")
