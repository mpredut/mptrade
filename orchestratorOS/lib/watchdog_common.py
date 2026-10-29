#!/usr/bin/env python3
"""watchdog_common.py — SHARED alerting infrastructure for the watchdogs
(watchdogfor_cache, watchdogfor_anomaly): loads the env, sends push (ntfy) + email,
and keeps the cooldown state. Extracted from watchdogfor_cache to avoid duplication.

Environment variables (from .env / config.env in the repository root):
  PHONE_ALERT_URL / NTFY_TOPIC                    — push channel
  SMTP_USERNAME / SMTP_PASSWORD / ALERT_TO_EMAIL  — email (optional)
  SMTP_SERVER / SMTP_PORT                         — email server (default gmail:587)
"""
import os
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent.parent  # orchestratorOS/lib/ -> repository root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from notify_engine.alertnotifiers import AlertNotifier  # noqa: E402
from botcore import (  # noqa: E402
    _parse_bool_flag,
    load_env_stack,
    required_bool_env,
    required_float_env,
    required_int_env,
)
from state_io import read_recovering_previous, save_json_state  # noqa: E402


def _watchdog_event(title, message):
    return {
        "type": "bot_event", "symbol": "SYSTEM", "name": title,
        "source": "watchdog", "body": message, "url": None,
    }


def load_env():
    """Load .env (gitignored secrets) + config.env (versioned) from the repository root."""
    load_env_stack(str(ROOT / ".env"))


def load_state(state_file):
    try:
        data = read_recovering_previous(state_file, json.loads, default=None)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_state(state_file, state):
    try:
        Path(state_file).parent.mkdir(parents=True, exist_ok=True)
        save_json_state(state_file, state)
    except Exception as e:
        print(f"[watchdog] cannot write state: {e}")


def _valid_webhook_url(raw_url: str) -> str | None:
    if not isinstance(raw_url, str) or not raw_url.strip():
        return None
    u = raw_url.strip()
    if u.startswith(("http://", "https://")):
        rest = u.split("://", 1)[1].rstrip("/")
        return u if "/" in rest and rest.split("/", 1)[1].strip() else None
    clean_topic = u.lstrip("/").strip()
    return f"https://ntfy.sh/{clean_topic}" if clean_topic else None


def send_ntfy(title, message):
    # watchdog = ERROR category -> prefer dedicated topic; fallback to PHONE_ALERT_URL
    topic = (os.environ.get("NTFY_TOPIC_ERROR") or "").split("#", 1)[0].strip()
    phone_url = (os.environ.get("PHONE_ALERT_URL") or "").split("#", 1)[0].strip()
    topic_url = _valid_webhook_url(topic) if topic else None
    fallback_url = _valid_webhook_url(phone_url) if phone_url else None
    url = topic_url or fallback_url or None
    if not url:
        print("[watchdog] no PHONE_ALERT_URL/NTFY_TOPIC_ERROR — skipping the push")
        return False
    ok = AlertNotifier.send_phone_webhook_batch([_watchdog_event(title, message)], webhook_url=url)
    print(f"[watchdog] push {'OK' if ok else 'FAILED'}")
    return ok


def alert(title, message):
    """Send the push to the ERROR topic; notify_engine mirrors that topic to email
    (policy: notify_engine/mailer.py), so no separate email call is made here."""
    return bool(send_ntfy(title, message))
