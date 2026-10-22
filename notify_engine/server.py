#!/usr/bin/env python3
import asyncio
import os
import sys
import json
import logging
import re
import time
import requests
from collections import defaultdict
from typing import Dict, Any, List

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
    from botcore import load_dotenv
    for fname in (".env", "config.env"):
        p = os.path.join(ROOT_DIR, fname)
        if os.path.isfile(p):
            load_dotenv(p)


def resolve_topic(category: str) -> str:
    from botcore import required_env
    cat = (category or "").strip().upper()
    if cat in ("SERVER", "SERVER_STATE", "SERVER_STATUS", "DEADMAN"):
        cat = "SERVER"
    key = f"NTFY_TOPIC_{cat}"
    if not os.environ.get(key, "").strip():
        _ensure_env_loaded()
    return required_env(key)

def _category_for_title_and_source(title: str, source: str) -> str:
    t = (title or "").upper()
    s = (source or "").lower()
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
    s = (source or "").lower()
    b = (bot_name or "").lower()
    if "kraken" in s or "kraken" in b or "xstock" in b:
        return "Kraken"
    if "hyperliquid" in s or "hl" in s or "hyperliquid" in b or b.startswith("hl"):
        return "Hyperliquid"
    if "t212" in s or "trading212" in s or "t212" in b:
        return "T212"
    if "binance" in s or "binance" in b or b in ("rtrade", "tradeall", "monitortrades", "order_retry", "monitororder") or "monitororder" in s:
        return "Binance"
    if "-" in bot_name:
        return bot_name.split("-")[0]
    return bot_name or (source.capitalize() if source else "")

class NotificationServer:
    def __init__(self):
        self._ensure_environment()
        self.ntfy_token = self._load_ntfy_token()
        self.rules = self._load_rules()
        self.cooldowns: Dict[str, float] = {}
        self.traceback_buffers: Dict[str, List[str]] = defaultdict(list)
        
        # Persistent Queue File
        self.queue_file = os.environ.get("NOTIFICATION_QUEUE_FILE") or os.path.join(ROOT_DIR, "cachedb", "notification_queue.jsonl")
        os.makedirs(os.path.dirname(self.queue_file), exist_ok=True)

    def _ensure_environment(self) -> None:
        from botcore import load_dotenv
        for fname in (".env", "config.env"):
            p = os.path.join(ROOT_DIR, fname)
            if os.path.isfile(p):
                load_dotenv(p)

    def _load_ntfy_token(self) -> str:
        tok = os.environ.get("NTFY_TOKEN", "").strip()
        if not tok:
            env_path = os.path.join(ROOT_DIR, ".env")
            if os.path.isfile(env_path):
                from botcore import load_dotenv
                load_dotenv(env_path)
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
            payload = {"__orchestrator_intent__": intent_type}
            payload.update(data)
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
        batch = pop_due_bulk_batch("ntfy", now=now, force=force)
        if not batch:
            return False
        bulk_title, bulk_body, bulk_prio, bulk_topic = batch
        if not bulk_topic:
            bulk_topic = self._resolve_topic("ERROR")
        prev = getattr(self, "_in_gated_dispatch", False)
        self._in_gated_dispatch = True
        try:
            return self._send_ntfy(bulk_title, bulk_body, bulk_prio, bulk_topic)
        finally:
            self._in_gated_dispatch = prev

    def _send_ntfy(self, title: str, message: str, priority: str, topic: str, is_retry: bool = False, skip_email: bool = False) -> bool:
        if not topic:
            return True
        if any(
            os.environ.get(k, "").strip().lower() in {"1", "true", "yes", "on"}
            for k in ("DISABLE_EXTERNAL_NOTIFICATIONS", "BACKTEST_MODE", "REPLAY_MODE")
        ):
            logging.debug(f"External notifications disabled: would send to {topic}: {title}")
            return True

        check_text = f"{title} {message} {topic}".upper()
        if any(fake in check_text for fake in ("ZZZFAKE", "FAKEUSD", "TESTPAIR", "TSTX", "FAKE_VENUE", "ZZZ")):
            logging.info(f"Skipping test/fake alert in _send_ntfy: {title}")
            return True

        if not is_retry and not getattr(self, "_in_gated_dispatch", False):
            from notify_engine.alertnotifiers import pass_common_gate
            allowed, reason, _, out_payload = pass_common_gate(
                "ntfy",
                title,
                message,
                priority,
                topic,
                urgent=(priority == "urgent"),
            )
            if not allowed:
                logging.info(f"Direct ntfy gated by common policy ({reason}): {title}")
                if (
                    reason not in ("duplicate", "similar_eliminated", "buffered_for_bulk")
                    and not skip_email
                    and is_email_mirrored(topic)
                ):
                    self._send_email(title, message)
                return False
            if out_payload:
                title, message, priority, topic = out_payload

        # ERROR / DEADMAN topics are mirrored to email (policy: notify_engine.mailer).
        # Sent before the push so an ntfy outage or quota never swallows the email.
        if not is_retry and not skip_email and is_email_mirrored(topic):
            self._send_email(title, message)
        
        url = f"https://ntfy.sh/{topic}"
        headers = {
            "Title": title.encode("utf-8").decode("latin-1", "ignore"),
            "Priority": priority
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
                    try:
                        time.sleep(float(retry_after))
                    except (ValueError, TypeError):
                        time.sleep(1)
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
            
    def _send_email(self, subject: str, message: str, is_retry: bool = False) -> bool:
        # Retries bypass dedup: the first attempt already reserved this fingerprint.
        ok = send_email(subject, message, dedup=not is_retry)
        if not ok and not is_retry and email_is_configured() and not last_error_is_auth_failure():
            self._enqueue_payload("email", {"subject": subject, "message": message})
        return ok

    @staticmethod
    def _format_price_alerts_batch_title(alerts: list) -> str:
        count = len(alerts)
        items = []
        symbols = []
        for a in alerts:
            sym = a.get("symbol", "N/A") if isinstance(a, dict) else getattr(a, "symbol", "N/A")
            atype = a.get("alert_type", "") if isinstance(a, dict) else getattr(a, "alert_type", "")
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
            sym = a.get("symbol", "N/A") if isinstance(a, dict) else getattr(a, "symbol", "N/A")
            if sym and sym != "N/A" and sym not in symbols:
                symbols.append(sym)
        syms_str = ", ".join(symbols)
        if syms_str and len(syms_str) <= 50:
            return f"New Coins ({count}): {syms_str}"
        else:
            return f"New Coins Discovered ({count})"

    def dispatch_alerts(self, alerts: list, webhook_url: str = None, bot_name: str = "") -> bool:
        if not alerts:
            return False

        first = alerts[0]
        # Resolve title, body, and source intelligently based on alert payload type
        if isinstance(first, dict):
            alert_type = first.get("type")
            if alert_type == "price_alert" or "alert_type" in first:
                if len(alerts) == 1:
                    sym = first.get("symbol", "N/A")
                    atype = first.get("alert_type", "alert")
                    pchg = float(first.get("percent_change", 0.0) or 0.0)
                    dir_str = "▲" if atype == "up" else "▼"
                    title = f"{sym} {dir_str} {pchg:+.2f}%"
                else:
                    title = self._format_price_alerts_batch_title(alerts)
                from notify_engine.alertnotifiers import AlertNotifier
                body = AlertNotifier.format_batch_message(alerts)
                source = first.get("source") or "price_alert"
            elif alert_type == "new_coin_discovered":
                if len(alerts) == 1:
                    sym = first.get("symbol", "N/A")
                    title = f"New Coin: {sym}"
                else:
                    title = self._format_new_coins_batch_title(alerts)
                from notify_engine.alertnotifiers import AlertNotifier
                body = AlertNotifier.format_batch_message(alerts)
                source = first.get("source") or "price_alert"
            elif alert_type == "bot_event":
                if len(alerts) == 1:
                    title = first.get("name", "Alert")
                    body = first.get("body", "")
                else:
                    title = f"{first.get('name', 'Alert')} ({len(alerts)})"
                    from notify_engine.alertnotifiers import AlertNotifier
                    body = AlertNotifier.format_batch_message(alerts)
                source = first.get("source", "")
            else:
                if len(alerts) == 1:
                    title = first.get("name") or first.get("title") or "Alert"
                    body = first.get("body") or first.get("message") or first.get("symbol") or ""
                else:
                    title = f"{first.get('name') or first.get('title') or 'Alert'} ({len(alerts)})"
                    from notify_engine.alertnotifiers import AlertNotifier
                    body = AlertNotifier.format_batch_message(alerts)
                source = first.get("source", "")
        elif isinstance(first, str):
            title = "Alert" if len(alerts) == 1 else f"Alerts ({len(alerts)})"
            body = "\n".join(alerts)
            source = "price_alert" if "price" in body.lower() else "system"
        else:
            if hasattr(first, "alert_type"):
                if len(alerts) == 1:
                    sym = getattr(first, "symbol", "N/A")
                    atype = getattr(first, "alert_type", "alert")
                    pchg = float(getattr(first, "percent_change", 0.0) or 0.0)
                    dir_str = "▲" if atype == "up" else "▼"
                    title = f"{sym} {dir_str} {pchg:+.2f}%"
                else:
                    title = self._format_price_alerts_batch_title(alerts)
                from notify_engine.alertnotifiers import AlertNotifier
                body = AlertNotifier.format_batch_message(alerts)
                source = getattr(first, "source", "price_alert")
            else:
                if len(alerts) == 1:
                    title = getattr(first, "name", getattr(first, "title", "Alert"))
                    body = getattr(first, "body", getattr(first, "message", getattr(first, "symbol", "")))
                else:
                    title = f"{getattr(first, 'name', getattr(first, 'title', 'Alert'))} ({len(alerts)})"
                    from notify_engine.alertnotifiers import AlertNotifier
                    body = AlertNotifier.format_batch_message(alerts)
                source = getattr(first, "source", "")

        topic = None
        if webhook_url:
            topic = webhook_url.rstrip("/").rsplit("/", 1)[-1]
        if not topic:
            topic = _topic_for_category(title, source)

        cat = _category_for_title_and_source(title, source)
        price_topic = self._resolve_topic("PRICE")
        trades_topic = self._resolve_topic("TRADES")

        # Strip any existing [price_notifier] prefix
        cleaned_title = re.sub(r"^\[price_notifier(?:\.py)?\]\s*", "", title, flags=re.IGNORECASE).strip()

        is_price = (
            topic == price_topic
            or cat == "PRICE"
            or bot_name.lower() in ("price_notifier", "price_notifier.py", "price_alert")
            or str(source).lower() in ("price_alert", "coinmarketcap", "coingecko", "dexscreener", "pricechecker", "notifier")
            or any(m in cleaned_title for m in ("▲", "▼"))
        )

        if is_price:
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
        all_syms = " ".join(str(a.get("symbol") if isinstance(a, dict) else getattr(a, "symbol", "") or "") for a in alerts)
        full_text = f"{full_title} {body} {all_syms}".upper()
        if any(fake in full_text for fake in ("ZZZFAKE", "FAKEUSD", "TESTPAIR", "TSTX", "FAKE_VENUE", "ZZZ")):
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

                intent = payload.get("__orchestrator_intent__")
                if intent == "ntfy_webhook":
                    # Direct retry payload
                    if "title" in payload:
                        self._gate_and_send_ntfy(
                            payload["title"],
                            payload["message"],
                            payload["priority"],
                            payload["topic"],
                            urgent=(payload.get("priority") == "urgent"),
                        )
                        return

                    # Standard alert formatting
                    alerts = payload.get("alerts", [])
                    webhook_url = payload.get("webhook_url")
                    self.dispatch_alerts(alerts, webhook_url=webhook_url, bot_name=bot_name)

                elif intent == "email":
                    # Same schema as the retry queue; rendered by AlertNotifier.send_email_batch.
                    self._send_email(payload.get("subject") or "Alert", payload.get("message") or "")
                return
            except json.JSONDecodeError as e:
                logging.warning(f"Failed to decode orchestrator intent JSON from {bot_name}: {e} (raw line: {line.strip()})")
                return
            except Exception as e:
                logging.error(f"Error processing orchestrator intent from {bot_name}: {e} (raw line: {line.strip()})", exc_info=True)
                return

        # 2. Check for the [NTFY] prefix shortcut
        if "[NTFY]" in line:
            if any(fake in line.upper() for fake in ("ZZZFAKE", "FAKEUSD", "TESTPAIR", "TSTX", "FAKE_VENUE", "ZZZ")):
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
        if any(fake in line.upper() for fake in ("ZZZFAKE", "FAKEUSD", "TESTPAIR", "TSTX", "FAKE_VENUE", "ZZZ")):
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
                last_time = self.cooldowns.get(rule_key, 0)
                cooldown_sec = rule.get("cooldown_minutes", 0) * 60
                
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
                failed_intents: set[str] = set()
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        payload = json.loads(line)
                    except Exception:
                        continue
                    
                    intent = payload.get("__orchestrator_intent__")
                    if intent not in failed_intents:
                        success = False
                        if intent == "ntfy_webhook":
                            if "title" in payload:
                                title = payload.get("title", "")
                                msg = payload.get("message", "")
                                if not title.startswith("[DELAYED]"):
                                    title = "[DELAYED] " + title
                                success = await asyncio.to_thread(
                                    self._send_ntfy,
                                    title,
                                    msg,
                                    payload.get("priority", "default"),
                                    payload.get("topic", "test-mptrade"),
                                    True
                                )
                            elif "alerts" in payload:
                                success = await asyncio.to_thread(
                                    self.dispatch_alerts,
                                    payload.get("alerts", []),
                                    payload.get("webhook_url"),
                                    ""
                                )
                            else:
                                success = True
                        elif intent == "email":
                            subj = payload.get("subject", "")
                            if not subj.startswith("[DELAYED]"):
                                subj = "[DELAYED] " + subj
                            success = await asyncio.to_thread(
                                self._send_email,
                                subj,
                                payload.get("message", ""),
                                True
                            )
                        else:
                            success = True  # Unknown intent, drop it
                            
                        if success:
                            success_count += 1
                        elif intent == "email" and last_error_is_auth_failure():
                            failed_intents.add(intent)
                            logging.warning(
                                f"Dropping queued email due to permanent SMTP authentication failure: {payload.get('subject', '')!r}"
                            )
                        else:
                            failed_intents.add(intent)
                            retries = int(payload.get("retry_count", 0)) + 1
                            if retries >= _MAX_QUEUE_RETRIES:
                                logging.warning(
                                    f"Dropping queued {intent} notification after {retries} failed retries"
                                )
                            else:
                                payload["retry_count"] = retries
                                remaining.append(json.dumps(payload))
                    elif intent == "email" and last_error_is_auth_failure():
                        logging.warning(
                            f"Dropping queued email due to permanent SMTP authentication failure: {payload.get('subject', '')!r}"
                        )
                    else:
                        remaining.append(line)
                        
                if success_count > 0 or len(remaining) != len(lines):
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
                    with open(self.queue_file, "w", encoding="utf-8") as f:
                        for r in remaining + newly_appended:
                            f.write(r + "\n")
                    if success_count > 0:
                        logging.info(f"Flushed {success_count} delayed notifications from queue.")

            except Exception as e:
                logging.error(f"Error flushing notification queue: {e}")
