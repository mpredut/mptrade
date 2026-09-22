# alert_notifiers.py
from __future__ import annotations

import requests
import hashlib
import json
import os
import smtplib
import subprocess
import sys
import time
from datetime import datetime, timezone
from email.mime.text import MIMEText
from pathlib import Path
from state_io import atomic_write_json
from typing import Any, Optional

# Import your modules
import log
from lock import FileLock

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


def _positive_int_env(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


def _delivery_state_path() -> Path:
    configured = os.environ.get("NOTIFICATION_STATE_FILE")
    return Path(configured).expanduser() if configured else BASE_DIR / "logs/notification_delivery_state.json"


def _load_delivery_state(path: Path, today: str) -> dict:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        state = {}
    if state.get("date_utc") != today:
        state = {"schema_version": 1, "date_utc": today, "channels": {}}
    state.setdefault("channels", {})
    return state


def _save_delivery_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        path, state, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )


def _alert_identity(alert: Any) -> dict:
    """Return stable identity excluding timestamps and prices that change each poll."""
    if isinstance(alert, dict):
        return {
            key: alert.get(key)
            for key in ("type", "symbol", "name", "source", "body", "url")
        }
    return {
        "type": alert.__class__.__name__,
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
        title = str(alert.get("name") or alert.get("symbol") or "").upper()
        source = str(alert.get("source") or "").lower()
        if any(marker in title for marker in _URGENT_MARKERS) or "watchdog" in source:
            return True
    return False


def _dedup_seconds(alerts: list[Any], urgent: bool) -> int:
    titles = " ".join(
        str(alert.get("name") or "").upper()
        for alert in alerts if isinstance(alert, dict)
    )
    if "DISPONIBIL" in titles:
        return _positive_int_env("NOTIFICATION_STARTUP_DEDUP_SECONDS", 6 * 60 * 60)
    if urgent:
        return _positive_int_env("NOTIFICATION_URGENT_DEDUP_SECONDS", 5 * 60)
    if any(
        not isinstance(alert, dict) or alert.get("type") == "new_coin_discovered"
        for alert in alerts
    ):
        return _positive_int_env("NOTIFICATION_PRICE_DEDUP_SECONDS", 30 * 60)
    return _positive_int_env("NOTIFICATION_DEDUP_SECONDS", 15 * 60)


def _reserve_delivery(channel: str, alerts: list[Any], *, urgent: bool) -> tuple[bool, str, bool]:
    """Atomically reserve one delivery across processes.

    Return ``(allowed, reason, warn_once)``. Only routine ntfy deliveries have a
    local daily cap. Urgent ntfy and all email bypass that cap, but still deduplicate.
    A network attempt counts conservatively because a timeout can follow acceptance.
    An actual ntfy provider quota remains binding and requires an email fallback.
    """
    now = time.time()
    today = datetime.fromtimestamp(now, timezone.utc).date().isoformat()
    path = _delivery_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = _delivery_fingerprint(alerts)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with FileLock(lock_path):
        state = _load_delivery_state(path, today)
        channel_state = state["channels"].setdefault(
            channel,
            {"sent": 0, "last": {}, "blocked": False, "budget_warning_sent": False},
        )
        if channel == "ntfy" and channel_state.get("blocked"):
            return False, "provider_daily_limit", False

        last = channel_state.setdefault("last", {})
        cutoff = now - 2 * 24 * 60 * 60
        channel_state["last"] = {
            key: value for key, value in last.items() if float(value) >= cutoff
        }
        previous = channel_state["last"].get(fingerprint)
        if previous is not None and now - float(previous) < _dedup_seconds(alerts, urgent):
            _save_delivery_state(path, state)
            return False, "duplicate", False

        sent = int(channel_state.get("sent", 0))
        if channel == "ntfy" and not urgent:
            # Preserve the existing routine allowance and provider-quota headroom.
            # The reserve no longer places a ceiling on urgent delivery attempts.
            budget = _positive_int_env("NTFY_DAILY_BUDGET", 100)
            reserve = _positive_int_env("NTFY_URGENT_RESERVE", 20)
            if sent >= max(0, budget - reserve):
                warn = not bool(channel_state.get("budget_warning_sent"))
                channel_state["budget_warning_sent"] = True
                _save_delivery_state(path, state)
                return False, "local_daily_budget", warn

        channel_state["sent"] = sent + 1
        channel_state["last"][fingerprint] = now
        _save_delivery_state(path, state)
        return True, "reserved", False


def _mark_provider_daily_limit(channel: str) -> bool:
    """Block the channel until UTC reset and request one alternate warning."""
    now = time.time()
    today = datetime.fromtimestamp(now, timezone.utc).date().isoformat()
    path = _delivery_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(path.with_suffix(path.suffix + ".lock")):
        state = _load_delivery_state(path, today)
        channel_state = state["channels"].setdefault(channel, {})
        first = not bool(channel_state.get("budget_warning_sent"))
        channel_state["blocked"] = True
        channel_state["budget_warning_sent"] = True
        _save_delivery_state(path, state)
        return first


def _is_provider_daily_limit(response: Any) -> bool:
    if getattr(response, "status_code", None) != 429:
        return False
    body = str(getattr(response, "text", "") or "").lower()
    return "42908" in body or "daily" in body


class AlertNotifier:

    def check_alert(condition, message, alert_interval=60):
        pass  # Placeholder for alert checking logic, can be implemented as needed
    
    @staticmethod
    def format_human_readable_time(value) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(value).strftime("%m-%d %H:%M:%S")
            except Exception:
                return str(value)
        if hasattr(value, "strftime"):
            try:
                return value.strftime("%m-%d %H:%M:%S")
            except Exception:
                return str(value)
        return str(value)

    @staticmethod
    def is_new_coin_alert(alert: Any) -> bool:
        return isinstance(alert, dict) and alert.get("type") == "new_coin_discovered"

    @staticmethod
    def alert_symbol(alert: Any) -> str:
        """Return the symbol from either a PriceAlert object or new-coin mapping."""
        if isinstance(alert, dict):
            return alert.get("symbol", "N/A")
        return getattr(alert, "symbol", "N/A")

    @staticmethod
    def utf8_header(value: str) -> str:
        """Encode an HTTP header while preserving non-ASCII symbols.

        Headers are Latin-1, while ntfy decodes this value as UTF-8, so pass UTF-8 bytes
        through Latin-1 to preserve the original symbol.
        """
        return value.encode("utf-8").decode("latin-1")

    @staticmethod
    def format_new_coin_message(alert: dict) -> str:
        lines = [
            f"🆕: {alert.get('symbol', 'N/A')} - {alert.get('name', alert.get('symbol', 'N/A'))}",
            f"Source: {alert.get('source', 'unknown')}",
            f"Added: {AlertNotifier.format_human_readable_time(alert.get('added_at'))}",
            f"Price: ${alert.get('price', 0):.4f}" if alert.get('price') is not None else "Price: N/A",
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
        body = (alert.get("body") or "").strip()
        head = body or alert.get("name", alert.get("symbol", "?"))
        parts = [head, str(alert.get("source", "?"))]
        ts = alert.get("added_at")
        if ts is not None:
            try:
                parts.append(ts.strftime("%m-%d %H:%M"))   # Omit the year.
            except Exception:  # noqa: BLE001
                pass
        return " · ".join(parts)

    @staticmethod
    def format_batch_message(alerts) -> str:
        # List comma-separated symbols on the first line.
        #symbols = ", ".join(alert.symbol for alert in alerts)
        #lines = [f"({len(alerts)}): {symbols}",    "",]
        lines = []
        for alert in alerts:
            if isinstance(alert, dict) and alert.get("type") == "bot_event":
                lines.append(AlertNotifier.format_bot_event(alert))
                continue
            if AlertNotifier.is_new_coin_alert(alert):
                lines.append(AlertNotifier.format_new_coin_message(alert))
                continue

            direction = "U" if alert.alert_type == "up" else "D"
            reference_time = AlertNotifier.format_human_readable_time(
                getattr(alert, "reference_time", None) or getattr(alert, "timestamp", None)
            )

            lines.append(
                f"{alert.symbol}: {direction} {alert.percent_change:+.2f}% "
                f"| C ${alert.current_price:.4f} | R ${alert.reference_price:.4f} "
                f"({reference_time})"
            )

            url = getattr(alert, "url", None)
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
        alert_file = Path(filename)
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
                pchg = getattr(alert, "percent_change", 0.0)
                cprice = getattr(alert, "current_price", 0.0)
                rprice = getattr(alert, "reference_price", 0.0)
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
    def send_email_batch(
        alerts: list[dict],
        recipient: Optional[str] = None,
        subject: Optional[str] = None,
    ) -> bool:
        if not alerts:
            return False
        import json
        from datetime import datetime
        def default_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return str(obj)
        intent = {
            "__orchestrator_intent__": "email",
            "subject": subject,
            "alerts": list(alerts)
        }
        print(json.dumps(intent, default=default_serializer), flush=True)
        return True

    @staticmethod
    def send_phone_webhook_batch(alerts, webhook_url: Optional[str] = None):
        if not alerts:
            return False
        if os.environ.get("MPTRADE_ORCHESTRATED") == "1":
            import json
            from datetime import datetime
            def default_serializer(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return str(obj)
            intent = {
                "__orchestrator_intent__": "ntfy_webhook",
                "webhook_url": webhook_url,
                "alerts": list(alerts)
            }
            print(json.dumps(intent, default=default_serializer), flush=True)
            return True
        else:
            from notify_engine.server import NotificationServer
            server = NotificationServer()
            return server.dispatch_alerts(list(alerts), webhook_url=webhook_url)

    @staticmethod
    def send(alert, enable_console=True, enable_file=True,
             enable_email=False, enable_phone_webhook=False, webhook_url=None):
        alerts = [alert] if not isinstance(alert, list) else alert
        if enable_console:
            for item in alerts:
                AlertNotifier.print_to_console(item)
        if enable_file:
            for item in alerts:
                AlertNotifier.save_to_file(item)
        if enable_email:
            AlertNotifier.send_email_batch(alerts)
        if enable_phone_webhook:
            AlertNotifier.send_phone_webhook_batch(alerts, webhook_url=webhook_url)

    @staticmethod
    def _send_urgent_email_fallback(alerts) -> None:
        pass

    @staticmethod
    def _send_budget_warning(channel: str, reason: str) -> None:
        pass

def notify(title: str, body: str, source: str, symbol: str,
           price: float = None, desktop: bool = False,
           email: bool = None) -> None:
    if os.environ.get("MPTRADE_ORCHESTRATED") == "1":
        import json
        intent = {
            "__orchestrator_intent__": "ntfy_webhook",
            "alerts": [{
                "type": "bot_event",
                "name": title,
                "body": body,
                "source": source,
                "symbol": symbol
            }]
        }
        print(json.dumps(intent), flush=True)
    else:
        from notify_engine.server import NotificationServer
        server = NotificationServer()
        server.dispatch_alerts([{
            "type": "bot_event",
            "name": title,
            "body": body,
            "source": source,
            "symbol": symbol
        }])


def bind_notify(symbol_env_keys: tuple, default_symbol: str):
    import os
    def bound_notify(title: str, body: str, source: str, price: float = None, desktop: bool = False, symbol: str = None, email: bool = None):
        resolved = symbol or next((os.environ[key] for key in symbol_env_keys if os.environ.get(key)), default_symbol)
        notify(title, body, source, resolved, price=price, desktop=desktop, email=email)
    bound_notify.__name__ = "notify"
    return bound_notify
