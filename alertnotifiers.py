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

BASE_DIR = Path(__file__).resolve().parent

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
                reference_time = AlertNotifier.format_human_readable_time(
                    getattr(alert, "reference_time", None) or getattr(alert, "timestamp", None)
                )
                f.write(f"[{datetime.now().isoformat()}] {alert.symbol} - {alert.alert_type} - {alert.percent_change:+.2f}%\n")
                f.write(f"  Price: ${alert.current_price:.4f}\n")
                f.write(f"  Reference: ${alert.reference_price:.4f} (at {reference_time})\n")
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
        alerts, email_config: Optional[dict] = None, subject: Optional[str] = None,
    ):
        email_config = email_config or {}
        smtp_server = email_config.get("smtp_server") or os.environ.get("SMTP_SERVER")
        smtp_port = email_config.get("smtp_port") or os.environ.get("SMTP_PORT")
        smtp_username = email_config.get("smtp_username") or os.environ.get("SMTP_USERNAME")
        smtp_password = email_config.get("smtp_password") or os.environ.get("SMTP_PASSWORD")
        to_email = email_config.get("to_email") or os.environ.get("ALERT_TO_EMAIL")

        if not alerts:
            print("[Notifier] Email: no alerts to send")
            return False

        if not smtp_server or not smtp_port or not smtp_username or not smtp_password or not to_email:
            print("[Notifier] Email: SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, and ALERT_TO_EMAIL are required")
            return False

        try:
            smtp_port = int(smtp_port)
        except ValueError:
            print("[Notifier] Email: SMTP_PORT must be an integer")
            return False

        alerts = list(alerts)
        urgent = _alerts_are_urgent(alerts)
        allowed, reason, _warn = _reserve_delivery("email", alerts, urgent=urgent)
        if not allowed:
            print(f"[Notifier] Email skipped by delivery policy: {reason}")
            return reason == "duplicate"

        symbols = ", ".join(AlertNotifier.alert_symbol(alert) for alert in alerts)
        subject = subject or f"CryptoAlerts: {len(alerts)} symbols ({symbols})"
        body = AlertNotifier.format_batch_message(alerts)
        msg = MIMEText(body, "plain", "utf-8")
        msg["From"] = smtp_username
        msg["To"] = to_email
        msg["Subject"] = subject

        try:
            with smtplib.SMTP(smtp_server, smtp_port, timeout=15) as server:
                server.starttls()
                server.login(smtp_username, smtp_password)
                server.sendmail(smtp_username, [to_email], msg.as_string())
            return True
        except Exception as e:
            print(f"[Notifier] Email batch exception: {e}")
            return False

    @staticmethod
    def send_phone_webhook_batch(alerts, webhook_url: Optional[str] = None):
        if not alerts:
            print("[Notifier] Phone webhook: no alerts to send")
            return False

        print(f"[Notifier] Phone webhook batch for {len(alerts)} alert(s)")
        if not webhook_url:
            print("[Notifier] Phone webhook: no target topic (webhook_url missing)")
            return False

        alerts = list(alerts)
        symbols = ", ".join(AlertNotifier.alert_symbol(alert) for alert in alerts)
        # For one bot event, use its action as the title; otherwise use ``(N): symbols``.
        if len(alerts) == 1 and isinstance(alerts[0], dict) and alerts[0].get("type") == "bot_event":
            title = alerts[0].get("name") or symbols
        else:
            title = f"({len(alerts)}): {symbols}"

        message = AlertNotifier.format_batch_message(alerts)
        tags = "chart_with_upwards_trend"
        payload = {"title": title, "message": message}

        is_ntfy = "ntfy.sh/" in webhook_url
        if is_ntfy:
            urgent = _alerts_are_urgent(alerts)
            allowed, reason, warn = _reserve_delivery("ntfy", alerts, urgent=urgent)
            if not allowed:
                print(f"[Notifier] ntfy skipped by delivery policy: {reason}")
                if warn:
                    AlertNotifier._send_budget_warning("ntfy", reason)
                if urgent and reason == "provider_daily_limit":
                    AlertNotifier._send_urgent_email_fallback(alerts)
                return reason == "duplicate"

        try:
            if is_ntfy:
                # ntfy decodes Title as UTF-8, preserving non-ASCII symbols.
                response = None
                for attempt in range(2):
                    _hdr = {
                        "Title": AlertNotifier.utf8_header(title),
                        "Priority": "urgent" if urgent else os.environ.get("NTFY_PRIORITY", "high"),
                        "Tags": "warning" if urgent else tags,
                    }
                    _tok = _ntfy_token()  # An ntfy.sh account -> a per-account quota, not a per-IP anonymous one.
                    if _tok:
                        _hdr["Authorization"] = f"Bearer {_tok}"
                    response = requests.post(
                        webhook_url,
                        data=message.encode("utf-8"),
                        headers=_hdr,
                        timeout=10,
                    )
                    if response.status_code != 429 or _is_provider_daily_limit(response):
                        break
                    if attempt == 0:
                        try:
                            wait = min(float(response.headers.get("Retry-After", 3) or 3), 8.0)
                        except (AttributeError, TypeError, ValueError):
                            wait = 3.0
                        print(f"[Notifier] a transient ntfy 429; retrying after {wait:.0f}s")
                        time.sleep(wait)
                # A stale account token must not break a topic that explicitly permits
                # anonymous publishing. Retry once without credentials; private topics
                # still reject the request, so their access control is not bypassed.
                if response.status_code in {401, 403} and _tok:
                    print("[Notifier] ntfy token rejected; retrying without credentials")
                    anonymous_headers = dict(_hdr)
                    anonymous_headers.pop("Authorization", None)
                    response = requests.post(
                        webhook_url,
                        data=message.encode("utf-8"),
                        headers=anonymous_headers,
                        timeout=10,
                    )
                provider_limited = _is_provider_daily_limit(response)
                if provider_limited:
                    if _mark_provider_daily_limit("ntfy"):
                        AlertNotifier._send_budget_warning("ntfy", "provider_daily_limit")
                if response.status_code >= 400:
                    print(f"[Notifier] ntfy batch error: {response.status_code} {response.text}")
                    if urgent and provider_limited:
                        AlertNotifier._send_urgent_email_fallback(alerts)
                    return False
                print(f"[Notifier] ntfy batch sent successfully for {len(alerts)} symbols")
                return True

            response = requests.post(webhook_url, json=payload, timeout=10)
            if response.status_code >= 400:
                print(f"[Notifier] Phone webhook batch error: {response.status_code} {response.text}")
                return False
            print(f"[Notifier] Phone webhook batch sent successfully for {len(alerts)} symbols")
            return True
        except Exception as e:
            print(f"[Notifier] Phone webhook batch exception: {e}")
            return False

    @staticmethod
    def _send_urgent_email_fallback(alerts) -> None:
        """Preserve the actual urgent incident when ntfy has exhausted its quota.

        Email keeps its own deduplication. Phone failure remains False even when
        SMTP accepts the fallback; SMTP acceptance is not proof of inbox receipt.

        The subject names the actual incident (bot-event name, or the symbols) so it
        is legible at a glance on a phone; the ntfy-quota note is trailing context,
        not the headline. The full incident stays in the body.
        """
        alerts = list(alerts)
        if len(alerts) == 1 and isinstance(alerts[0], dict) and alerts[0].get("name"):
            headline = str(alerts[0]["name"])
        else:
            symbols = ", ".join(dict.fromkeys(
                AlertNotifier.alert_symbol(a) for a in alerts)) or "?"
            headline = f"{len(alerts)} alert(s): {symbols}"
        if len(headline) > 90:
            headline = headline[:87] + "..."
        subject = f"Urgent: {headline} (ntfy quota exhausted -> email)"
        sent = AlertNotifier.send_email_batch(alerts, subject=subject)
        print(f"[Notifier] Urgent email fallback {'accepted' if sent else 'failed'}")

    @staticmethod
    def _send_budget_warning(channel: str, reason: str) -> None:
        """Send one email fallback rather than compensating for every suppressed message."""
        if reason == "local_daily_budget":
            body = (
                f"Routine {channel} notifications reached the local daily allowance. "
                "Urgent ntfy alerts and all email remain exempt from local volume caps."
            )
        else:
            body = (
                f"Channel {channel} reached its provider daily quota. "
                "Urgent incidents use email fallback, subject to SMTP availability."
            )
        alert = {
            "type": "bot_event", "symbol": "SYSTEM",
            "name": f"ERROR: {channel.upper()} notification limit", "source": "notifier",
            "body": body,
            "added_at": datetime.now(),
        }
        AlertNotifier.send_email_batch(
            [alert], subject=f"Trading alerts: {channel} limit reached",
        )


    #Combined handler that sends alerts through multiple channels.    
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


def _topic_for(title: str, source: str) -> Optional[str]:
    """Route a notification to the ntfy topic selected by title and source.

    ``guard`` covers stop-loss, trailing, crash, and liquidation events where money is
    at risk. ``error`` covers failures, manual action, missing positions, and watchdog
    events. ``price`` covers price-threshold alerts. ``trades`` covers fills, available
    balances, routine delta-neutral open/close/funding events, and everything else.
    Read ``NTFY_TOPIC_<CATEGORY>``. This matches the
    email policy, where guard and error events are urgent.
    """
    t = title.upper(); s = (source or "").lower()
    if any(m in t for m in _GUARD_MARKERS):
        cat = "GUARD"
    elif any(m in t for m in _OPS_MARKERS) or "watchdog" in s:
        cat = "ERROR"
    elif "alert" in s or "prag" in t.lower() or "threshold" in t.lower():
        cat = "PRICE"
    else:
        cat = "TRADES"       # Include routine DN events; they do not have a separate topic.
    return os.environ.get(f"NTFY_TOPIC_{cat}")
# Do not include 📉, which is also used by informational loss alerts such as ``📉 SPCX -8%``,
# or a lone ⚠, which is too broad. ``TRAILING`` identifies trailing events. Urgent DN events
# also include LIQUID/ERROR/MANUAL in their titles (or their Romanian equivalents, see the
# marker tuples). Override with email=True/False when needed.


_NTFY_TOKEN_CACHE = None


def _ntfy_token():
    """Token ntfy.sh: env NTFY_TOKEN, altfel ~/.binance_ntfy_token (600). Cota per-cont."""
    global _NTFY_TOKEN_CACHE
    if _NTFY_TOKEN_CACHE is not None:
        return _NTFY_TOKEN_CACHE
    tok = os.environ.get("NTFY_TOKEN", "").strip()
    if not tok:
        try:
            with open(os.path.expanduser("~/.binance_ntfy_token"), encoding="utf-8") as fh:
                tok = fh.read().strip()
        except OSError:
            tok = ""
    _NTFY_TOKEN_CACHE = tok
    return tok


def notify(title: str, body: str, source: str, symbol: str,
           price: Optional[float] = None, desktop: bool = False,
           email: Optional[bool] = None) -> None:
    """Shared notification wrapper used by Binance, Kraken, HL, and T212 bots.

    Send a terminal bell, an ntfy alert, an optional email, and an optional desktop
    notification through ``AlertNotifier``. The caller resolves ``symbol`` as the asset
    label. Notification failures never interrupt trading. ``print`` works both in the
    fleet, where log.py captures it, and in standalone bots whose stdout goes to their log.

    By default, urgent stop-loss, trailing, crash, liquidation, error, and watchdog
    events also use email. Informational fills, available-balance messages, and price
    alerts use ntfy only unless explicitly overridden. This is shared by all venues.
    """
    # Tests and offline replays can exercise the same paths as live execution.
    # This central kill switch prevents external effects if a test forgets to inject
    # a fake ntfy/email/desktop/beep notifier.
    if os.environ.get("DISABLE_EXTERNAL_NOTIFICATIONS", "").strip().lower() in {
        "1", "true", "yes", "on",
    }:
        return

    for _ in range(5):
        sys.stdout.write("\a")
        sys.stdout.flush()
        time.sleep(0.2)
    alert = {
        "type": "bot_event",          # Bot trade/guard event with compact rendering.
        "symbol": symbol,             # Do not use the new-coin discovery template.
        "name": title,                # The action becomes the ntfy title.
        "source": source,
        "price": price,
        "body": body,                 # Compact one-line detail built by the caller.
        "added_at": datetime.now(),
        "url": None,
    }
    ntfy_topic = _topic_for(title, source)          # Route by category.
    ntfy_url = f"https://ntfy.sh/{ntfy_topic}" if ntfy_topic else None
    try:
        AlertNotifier.send_phone_webhook_batch([alert], webhook_url=ntfy_url)
    except Exception as e:  # noqa: BLE001
        print(f"  ! notify ntfy failed: {e}")
    if email is None:   # Use the same urgency as ntfy, including watchdog-source incidents.
        email = _alerts_are_urgent([alert])
    if email and os.environ.get("ALERT_TO_EMAIL"):
        try:
            AlertNotifier.send_email_batch([alert])
        except Exception as e:  # noqa: BLE001
            print(f"  ! notify email failed: {e}")
    if desktop:
        try:
            subprocess.run(["notify-send", "-u", "critical", title, body], check=False)
        except (FileNotFoundError, OSError):
            pass


def bind_notify(symbol_env_keys: tuple[str, ...], default_symbol: str):
    """Bind the shared notifier to a venue's symbol convention.

    The wrapper preserves the Kraken/HL/T212 signature and accepts an explicit symbol
    for multi-asset processes. Venue files remain import shims rather than duplicate
    symbol-resolution implementations.
    """
    if not symbol_env_keys:
        raise ValueError("bind_notify requires at least one environment key")

    def bound_notify(
        title: str,
        body: str,
        source: str,
        price: Optional[float] = None,
        desktop: bool = False,
        symbol: Optional[str] = None,
        email: Optional[bool] = None,
    ) -> None:
        resolved = symbol or next(
            (os.environ[key] for key in symbol_env_keys if os.environ.get(key)),
            default_symbol,
        )
        notify(
            title, body, source, resolved,
            price=price, desktop=desktop, email=email,
        )

    bound_notify.__name__ = "notify"
    return bound_notify
