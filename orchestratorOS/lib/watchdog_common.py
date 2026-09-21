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
    load_env_stack, required_bool_env, required_float_env, required_int_env,
)
from state_io import atomic_write_json  # noqa: E402


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
        return json.load(open(state_file))
    except Exception:
        return {}


def save_state(state_file, state):
    try:
        atomic_write_json(state_file, state)
    except Exception as e:
        print(f"[watchdog] cannot write state: {e}")


def send_ntfy(title, message):
    # watchdog = ERROR category -> prefer dedicated topic; fallback to PHONE_ALERT_URL
    topic = os.environ.get("NTFY_TOPIC_ERROR")
    url = (f"https://ntfy.sh/{topic}" if topic else None) or os.environ.get("PHONE_ALERT_URL")
    if not url:
        print("[watchdog] no PHONE_ALERT_URL/NTFY_TOPIC_ERROR — skipping the push")
        return False
    ok = AlertNotifier.send_phone_webhook_batch([_watchdog_event(title, message)], webhook_url=url)
    print(f"[watchdog] push {'OK' if ok else 'FAILED'}")
    return ok


def send_email(subject, body):
    ok = AlertNotifier.send_email_batch(
        [_watchdog_event(subject, body)], subject=subject,
    )
    print(f"[watchdog] email {'sent' if ok else 'skipped/failed'}")
    return ok


def alert(title, message):
    """Send push + email. Returns True if at least one of them succeeded."""
    ok_push = send_ntfy(title, message)
    ok_mail = send_email(title, message)
    return bool(ok_push or ok_mail)
