#!/usr/bin/env python3
"""mailer.py — the single SMTP choke point for every email the fleet sends.

Owns two things and nothing else:
  * Delivery: ``send_email(subject, body)`` over SMTP (STARTTLS), with cross-process
    deduplication through the shared delivery policy in ``alertnotifiers``.
  * Mirror policy: which ntfy topics are ALSO copied to email. The ERROR channel and
    the DEADMAN ("server down") channel are mirrored; everything else is push-only
    unless a caller explicitly asks for email.

Environment (``.env`` / ``config.env`` in the repository root):
  SMTP_USERNAME / SMTP_PASSWORD / ALERT_TO_EMAIL  — required to send
  SMTP_SERVER / SMTP_PORT                         — default smtp.gmail.com:587
  DISABLE_EXTERNAL_NOTIFICATIONS                  — global kill switch

CLI (used by the bash notifiers in orchestratorOS/):
  mailer.py [--topic TOPIC] SUBJECT BODY
With ``--topic`` the email is sent only if that topic is mirrored, so the policy
stays here and the shell scripts never duplicate it.
"""
from __future__ import annotations

import logging
import os
import smtplib
import sys
from email.mime.text import MIMEText

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Categories whose ntfy topic (NTFY_TOPIC_<CATEGORY>) is mirrored to email.
EMAIL_MIRRORED_CATEGORIES = ("ERROR", "DEADMAN", "SERVER")

_SMTP_TIMEOUT_SEC = 15


def _disabled() -> bool:
    return os.environ.get("DISABLE_EXTERNAL_NOTIFICATIONS", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def load_env() -> None:
    """Load config.env (versioned) and .env (secrets) without overriding the process env."""
    from botcore import load_dotenv
    for fname in ("config.env", ".env"):
        path = os.path.join(ROOT_DIR, fname)
        if os.path.isfile(path):
            load_dotenv(path)


def email_mirrored_topics() -> set[str]:
    topics = (os.environ.get(f"NTFY_TOPIC_{cat}", "").strip() for cat in EMAIL_MIRRORED_CATEGORIES)
    return {t for t in topics if t}


def is_email_mirrored(topic: str | None) -> bool:
    return bool(topic) and topic in email_mirrored_topics()


def _config() -> dict | None:
    if not os.environ.get("SMTP_USERNAME") or not os.environ.get("ALERT_TO_EMAIL"):
        load_env()
    cfg = {
        "server": os.environ.get("SMTP_SERVER", "").strip() or "smtp.gmail.com",
        "port": os.environ.get("SMTP_PORT", "").strip() or "587",
        "username": os.environ.get("SMTP_USERNAME", "").strip(),
        "password": os.environ.get("SMTP_PASSWORD", "").strip(),
        "to": os.environ.get("ALERT_TO_EMAIL", "").strip(),
    }
    missing = [k for k in ("username", "password", "to") if not cfg[k]]
    if missing:
        logging.warning(f"Email not configured (missing: {', '.join(missing)})")
        return None
    try:
        cfg["port"] = int(cfg["port"])
    except ValueError:
        logging.warning(f"Email not configured (SMTP_PORT is not an integer: {cfg['port']!r})")
        return None
    if "gmail" in cfg["server"]:
        # Google shows app passwords in 4-letter groups; the spaces are not part of it.
        cfg["password"] = cfg["password"].replace(" ", "")
    return cfg


def is_configured() -> bool:
    return _config() is not None


def _smtp_send(msg: MIMEText, cfg: dict) -> None:
    with smtplib.SMTP(cfg["server"], cfg["port"], timeout=_SMTP_TIMEOUT_SEC) as smtp:
        smtp.starttls()
        smtp.login(cfg["username"], cfg["password"])
        smtp.sendmail(cfg["username"], [cfg["to"]], msg.as_string())


def send_email(subject: str, body: str, *, dedup: bool = True) -> bool:
    """Send one email. Returns True on SMTP acceptance (or a suppressed duplicate)."""
    if _disabled():
        return True
    cfg = _config()
    if cfg is None:
        return False
    subject = (subject or "Alert").strip() or "Alert"
    body = body or ""
    if dedup:
        from notify_engine.alertnotifiers import _reserve_delivery
        identity = [{"type": "email", "name": subject, "body": body, "source": "mailer"}]
        allowed, reason, _ = _reserve_delivery("email", identity, urgent=True)
        if not allowed:
            logging.info(f"Email skipped by delivery policy ({reason}): {subject}")
            return reason == "duplicate"
    msg = MIMEText(body, "plain", "utf-8")
    msg["From"] = cfg["username"]
    msg["To"] = cfg["to"]
    msg["Subject"] = subject
    try:
        _smtp_send(msg, cfg)
    except Exception as e:  # noqa: BLE001 — any SMTP/network failure is a soft failure
        logging.error(f"Email delivery failed for {subject!r}: {e}")
        return False
    logging.info(f"Dispatched email to {cfg['to']}: {subject}")
    return True


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Send an alert email (SMTP).")
    parser.add_argument("--topic", help="Only send if this ntfy topic is email-mirrored.")
    parser.add_argument("subject")
    parser.add_argument("body", nargs="?", default="")
    args = parser.parse_args(argv)
    load_env()
    if args.topic is not None and not is_email_mirrored(args.topic):
        return 0
    return 0 if send_email(args.subject, args.body) else 1


if __name__ == "__main__":
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] mailer: %(message)s")
    sys.exit(main())
