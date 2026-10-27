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
_LAST_ERROR_IS_AUTH = False


def last_error_is_auth_failure() -> bool:
    return _LAST_ERROR_IS_AUTH


def _disabled() -> bool:
    return any(
        os.environ.get(k, "").strip().lower() in {"1", "true", "yes", "on"}
        for k in ("DISABLE_EXTERNAL_NOTIFICATIONS", "BACKTEST_MODE", "REPLAY_MODE")
    )


def load_env() -> None:
    """Load config.env (versioned) and .env (secrets) without overriding the process env."""
    from botcore import load_dotenv
    for fname in ("config.env", ".env"):
        path = os.path.join(ROOT_DIR, fname)
        if os.path.isfile(path):
            load_dotenv(path)


def email_mirrored_topics() -> set[str]:
    if not os.environ.get("NTFY_TOPIC_ERROR"):
        load_env()
    topics = {
        os.environ.get(f"NTFY_TOPIC_{cat}", "").split("#", 1)[0].strip()
        for cat in EMAIL_MIRRORED_CATEGORIES
    }
    topics.discard("")
    if not topics:
        topics.update({"ntfy-error-1978", "ntfy-server-1978"})
    return topics


def is_email_mirrored(topic: str | None) -> bool:
    if not topic or not isinstance(topic, str):
        return False
    t_clean = topic.strip().rstrip("/")
    if "/" in t_clean:
        t_clean = t_clean.rsplit("/", 1)[-1]
    return bool(t_clean) and t_clean in email_mirrored_topics()


def _config() -> dict | None:
    if not os.environ.get("SMTP_USERNAME") or not os.environ.get("ALERT_TO_EMAIL"):
        load_env()
    cfg = {
        "server": os.environ.get("SMTP_SERVER", "").split("#", 1)[0].strip() or "smtp.gmail.com",
        "port": os.environ.get("SMTP_PORT", "").split("#", 1)[0].strip() or "587",
        "username": os.environ.get("SMTP_USERNAME", "").split("#", 1)[0].strip(),
        "password": os.environ.get("SMTP_PASSWORD", "").strip(),
        "to": os.environ.get("ALERT_TO_EMAIL", "").split("#", 1)[0].strip(),
    }
    missing = [k for k in ("username", "password", "to") if not cfg[k]]
    if missing:
        logging.warning(f"Email not configured (missing: {', '.join(missing)})")
        return None
    try:
        port_f = float(cfg["port"])
        if not port_f.is_integer():
            raise ValueError("port must be integer")
        port_i = int(port_f)
        if not (1 <= port_i <= 65535):
            raise ValueError("port out of range")
        cfg["port"] = port_i
    except (ValueError, OverflowError):
        logging.warning(f"Email not configured (SMTP_PORT is not a valid TCP port: {cfg['port']!r})")
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
        smtp.ehlo()
        if "PLAIN" in smtp.esmtp_features.get("auth", ""):
            smtp.esmtp_features["auth"] = "PLAIN"
        smtp.login(cfg["username"], cfg["password"])
        smtp.sendmail(cfg["username"], [cfg["to"]], msg.as_string())


def send_email(subject: str, body: str, *, dedup: bool = True) -> bool:
    """Send one email. Returns True on SMTP acceptance (or a suppressed duplicate)."""
    global _LAST_ERROR_IS_AUTH
    _LAST_ERROR_IS_AUTH = False
    if _disabled():
        return True
    cfg = _config()
    if cfg is None:
        return False
    subj_str = "" if isinstance(subject, bool) else str(subject or "")
    subject = subj_str.strip() or "Alert"
    body = "" if isinstance(body, bool) else str(body or "")
    dedup_flag = (
        dedup
        if isinstance(dedup, bool)
        else (str(dedup).strip().lower() in {"1", "true", "yes", "on"} if isinstance(dedup, str) else bool(dedup))
    )
    if dedup_flag:
        from notify_engine.alertnotifiers import pass_common_gate
        identity = [{"type": "email", "name": subject, "body": body, "source": "mailer"}]
        allowed, reason, _, out_payload = pass_common_gate(
            "email", subject, body, "high", "email", identity, urgent=True,
        )
        if not allowed:
            logging.info(f"Email gated by common delivery policy ({reason}): {subject}")
            return reason in ("duplicate", "similar_eliminated", "buffered_for_bulk")
        if out_payload:
            subject, body, _, _ = out_payload
    msg = MIMEText(body, "plain", "utf-8")
    msg["From"] = cfg["username"]
    msg["To"] = cfg["to"]
    msg["Subject"] = subject
    try:
        _smtp_send(msg, cfg)
    except smtplib.SMTPAuthenticationError as e:
        _LAST_ERROR_IS_AUTH = True
        logging.warning(f"Email delivery skipped for {subject!r} (SMTP auth rejected): {e}")
        return False
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
