#!/usr/bin/env python3
"""watchdog_common.py -- SHARED alerting infrastructure for the watchdogs
(watchdogfor_cache, watchdogfor_anomaly): loads the env, sends push (ntfy) + email,
and keeps the cooldown state. Extracted from watchdogfor_cache to avoid duplication.

Environment variables (from .env / config.env in the repository root):
  PHONE_ALERT_URL / NTFY_TOPIC                    -- push channel
  SMTP_USERNAME / SMTP_PASSWORD / ALERT_TO_EMAIL  -- email (optional)
  SMTP_SERVER / SMTP_PORT                         -- email server (default gmail:587)
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent.parent  # orchestratorOS/lib/ -> repository root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from botcore import optional_env  # noqa: E402
from notify_engine.alertnotifiers import (  # noqa: E402
    AlertNotifier,
    format_ntfy_url,
    make_bot_event_alert,
)
from state_io import read_json_dict, save_json_state  # noqa: E402


def load_state(state_file):
    return read_json_dict(state_file, default={}, recover_previous=True)


def save_state(state_file, state):
    try:
        Path(state_file).parent.mkdir(parents=True, exist_ok=True)
        save_json_state(state_file, state)
    except Exception as e:
        print(f"[watchdog] cannot write state: {e}")


def send_ntfy(title, message):
    # watchdog = ERROR category -> prefer dedicated topic; fallback to PHONE_ALERT_URL
    url = (
        format_ntfy_url(optional_env("NTFY_TOPIC_ERROR"))
        or format_ntfy_url(optional_env("PHONE_ALERT_URL"))
        or None
    )
    if not url:
        print("[watchdog] " + "no PHONE_ALERT_URL/NTFY_TOPIC_ERROR — skipping the push")
        return False
    event = make_bot_event_alert(title, message, "watchdog", url=None)
    ok = AlertNotifier.send_phone_webhook_batch([event], webhook_url=url)
    print(f"[watchdog] push {'OK' if ok else 'FAILED'}")
    return bool(ok)
