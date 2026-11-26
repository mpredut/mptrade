"""notifyapi.py -- public notification API, formatting, topic routing, and email-mirror policy.

Bots, guards, and CLI scripts call ``notify()``, ``bind_notify()``, ``emit_trade_notification()``,
``dispatch_order_event_notification()``, or ``AlertNotifier`` here. Delivery rate-limiting and
deduplication live in ``notify_engine.ntfypolicy``; HTTP ntfy transport, log-rule parsing, and
retry queuing live in ``notify_engine.ntfydispatcher``; SMTP lives in ``notify_engine.ntfymailer``.
"""
from __future__ import annotations

import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from botcore import (
    _parse_bool_flag,
    bool_env,
    clean_order_id_str,
    exact_pos_int,
    format_notify_price,
    format_notify_qty,
    is_replay_or_backtest_env,
    load_env_stack,
    load_local_module,
    normalize_epoch_sec,
    obj_attr,
    optional_env,
    required_env,
    safe_float,
    safe_pos_float,
    split_csv,
)
from cachepaths import (
    canonical_base_asset,
    clean_non_bool_str,
    clean_scalar_str,
    clean_symbol_key,
    clean_upper_scalar,
    has_equity_suffix,
    is_equity_venue_or_suffix,
    is_kraken_tokenized_asset,
    normalize_venue_name,
    split_symbol_assets,
    venue_default_quote_asset,
)
from notify_engine.ntfypolicy import (
    _AUTO_TOPIC_TAG_RE,
    _TRADE_MARKERS,
    _alerts_are_urgent,
    _canonical_topic_key,
    make_bot_event_alert,
    sanitize_single_line,
    strip_owner_tags,
)
from state_io import encode_json_payload

BASE_DIR = Path(__file__).resolve().parent.parent

_FAKE_ALERT_MARKERS = (
    "ZZZFAKE", "FAKEUSD", "TESTPAIR", "TSTX", "FAKE_VENUE", "ZZZ",
)

NOTIFY_DETAIL_SEP = " \u00b7 "


def _clean_notify_scalar(val: Any) -> str:
    return clean_scalar_str(clean_non_bool_str(val)) or ""


def format_owner_tags(
    owner: Any,
    *,
    ignore: tuple[str, ...] = ("unspecified", "unknown", "order_retry"),
) -> tuple[str, str]:
    """Return ``(owner_tag, owner_body)`` as ``(' [owner]', ' (owner)')`` when valid, else ``('', '')``."""
    s = clean_non_bool_str(owner).strip("[]()").strip()
    if not s or s.lower() in ignore or s.lower().startswith("alt:"):
        return "", ""
    return f" [{s}]", f" ({s})"


def format_trade_venue_tag(
    venue: Any = "binance",
    owner: Any = None,
    *,
    symbol: Any = None,
    ignore: tuple[str, ...] = ("unspecified", "unknown", "order_retry"),
) -> str:
    """Return ``'[venue:owner]'`` when owner is valid, else ``'[venue]'``."""
    v = normalize_venue_name(venue, default="") or ""
    _, owner_body = format_owner_tags(owner, ignore=ignore)
    o = owner_body.strip(" ()").lower() if owner_body else ""
    if o in ("ai_executor", "auto_trade_executor"):
        is_eq = is_equity_venue_or_suffix(symbol, v) or has_equity_suffix(symbol, include_compact=True)
        o = "auto_sharetrade_executor" if is_eq else "auto_cryptotrade_executor"
    if v and o:
        return f"[{v}:{o}]"
    if v:
        return f"[{v}]"
    return f"[{o}]" if o else ""


def format_trade_notification(
    status: str,
    side: Any,
    symbol: Any,
    price_str: Any,
    qty: Any,
    val: Any,
    quote: Any = None,
    order_id: Any = None,
    *,
    venue: Any = "binance",
    owner: Any = None,
    fills_count: int = 1,
    cum_qty: Optional[float] = None,
    cum_val: Optional[float] = None,
    cum_fills: int = 0,
    detail: Any = None,
    thesis: Any = None,
    prefix: Any = None,
) -> tuple[str, str]:
    """Build canonical ``(title, body)`` for order/trade notifications:
    Title: ``<STATUS> <SIDE> <BASE>@<PRICE> ~<TOTAL> $~ [venue:module]``
    Body:  ``~<VAL> <QUOTE> [| detail] (qty: <QTY>, order: <ID>)``
    """
    st_s = clean_upper_scalar(clean_non_bool_str(status)) or "FILLED"
    side_s = clean_upper_scalar(clean_non_bool_str(side)) or "ORDER"
    raw_sym = _clean_notify_scalar(symbol) or "TRADE"
    sym_parts = split_symbol_assets(raw_sym)
    if clean_symbol_key(raw_sym) == "USDCUSDT":
        sym_short = "USDCUSDT"
    else:
        sym_short = canonical_base_asset(raw_sym) or (sym_parts[0] if sym_parts else raw_sym.upper())

    v_norm = normalize_venue_name(venue, default="")
    raw_quote = _clean_notify_scalar(quote)
    if raw_quote:
        q_ccy = raw_quote.upper()
    elif sym_parts and sym_parts[1]:
        q_ccy = sym_parts[1].upper()
    else:
        q_ccy = venue_default_quote_asset(v_norm)

    if isinstance(price_str, (int, float)) and not isinstance(price_str, bool):
        px_f_in = safe_pos_float(price_str, default=0.0) or 0.0
        px_clean = format_notify_price(px_f_in, comma=False) if px_f_in > 0.0 else ""
    else:
        px_raw = _clean_notify_scalar(price_str).lstrip("$").strip()
        px_num_chk = safe_pos_float(px_raw.replace(",", "").lstrip("~"), default=None)
        px_clean = "" if (not px_raw or (px_num_chk is None and px_raw in ("0", "0.0", "0.00"))) else px_raw

    qf = safe_pos_float(qty, default=0.0) or 0.0
    vf = safe_pos_float(val, default=None)
    if vf is None:
        px_num = safe_pos_float(px_clean.replace(",", "").lstrip("~"), default=0.0) or 0.0
        vf = px_num * qf

    ccy_sym = "\u20ac" if q_ccy == "EUR" else "$"
    if vf >= 10.0:
        tot_subj = f"~{math.floor(vf + 0.5)} {ccy_sym}~"
    elif vf > 0.0:
        tot_subj = f"~{vf:.2f} {ccy_sym}~"
    else:
        tot_subj = ""

    sym_px = f"{sym_short}@{px_clean}" if px_clean else sym_short
    vtag = format_trade_venue_tag(venue, owner, symbol=raw_sym)
    pfx_s = _clean_notify_scalar(prefix)
    title_parts = [p for p in (pfx_s, st_s, side_s, sym_px, tot_subj, vtag) if p]
    title = " ".join(title_parts)

    detail_s = (
        NOTIFY_DETAIL_SEP.join(p for item in detail if (p := _clean_notify_scalar(item)))
        if isinstance(detail, (list, tuple))
        else _clean_notify_scalar(detail)
    )
    head_parts: list[str] = []
    if vf > 0.0 or not detail_s:
        head_parts.append(f"~{vf:.2f} {q_ccy}")
    if fills_count > 1:
        head_parts.append(f"{fills_count} fills")
    head_line = NOTIFY_DETAIL_SEP.join(head_parts)
    if cum_fills > fills_count and cum_qty is not None and cum_val is not None:
        head_line = f"{head_line} [total: ~{cum_val:.2f} {q_ccy}, {cum_fills} fills]".strip()
    if detail_s:
        head_line = f"{head_line}{NOTIFY_DETAIL_SEP}{detail_s}" if head_line else detail_s

    oid_clean = clean_order_id_str(order_id)
    has_oid = bool(oid_clean)
    tail_items: list[str] = []
    if qf > 0.0 or (has_oid and st_s in ("PLACED", "FILLED", "EXPIRED")):
        tail_items.append(f"qty: {format_notify_qty(qf)}")
    if cum_fills > fills_count and cum_qty is not None:
        tail_items.append(f"total_qty: {format_notify_qty(cum_qty)}")
    if has_oid:
        tail_items.append(f"order: {oid_clean}")
    tail_str = f"({', '.join(tail_items)})" if tail_items else ""

    thesis_s = _clean_notify_scalar(thesis)
    if thesis_s:
        lines = [head_line, f"Thesis: {thesis_s}"]
        if tail_str:
            lines.append(tail_str)
        body = "\n".join(ln for ln in lines if ln)
    else:
        body = f"{head_line} {tail_str}".strip() if tail_str else head_line
    return title, body


def emit_trade_notification(
    notify_fn: Any,
    status: str,
    side: Any,
    symbol: Any,
    price_str: Any,
    qty: Any = 0.0,
    val: Any = None,
    quote: Any = None,
    order_id: Any = None,
    *,
    notify_kwargs: Optional[dict[str, Any]] = None,
    **fmt_kwargs: Any,
) -> tuple[str, str]:
    """Format and emit a canonical order/trade notification via ``notify_fn`` when callable."""
    title, body = format_trade_notification(
        status, side, symbol, price_str, qty, val, quote, order_id, **fmt_kwargs
    )
    if callable(notify_fn):
        notify_fn(title=title, body=body, **(notify_kwargs or {}))
    return title, body


def is_fake_alert_text(*parts: Any) -> bool:
    """Return True when any part contains synthetic/test instrument markers."""
    joined = " ".join("" if isinstance(p, bool) or p is None else str(p) for p in parts).upper()
    return any(marker in joined for marker in _FAKE_ALERT_MARKERS)


def format_ntfy_url(topic: Any) -> str:
    """Format an ntfy topic name or full HTTP(S) URL into a canonical ntfy endpoint URL."""
    if not isinstance(topic, str) or isinstance(topic, bool):
        return ""
    clean = topic.split("#", 1)[0].strip()
    if not clean:
        return ""
    if clean.startswith(("http://", "https://")):
        rest = clean.split("://", 1)[1].rstrip("/")
        return clean if "/" in rest and rest.split("/", 1)[1].strip() else ""
    clean_topic = clean.lstrip("/")
    return f"https://ntfy.sh/{clean_topic}" if clean_topic else ""


def _emit_orchestrator_intent(payload: dict[str, Any], *, default: Any = None) -> bool:
    sys.stdout.write(encode_json_payload(payload, default=default) + "\n")
    sys.stdout.flush()
    return True


class AlertNotifier:

    @staticmethod
    def format_human_readable_time(value, fmt: str = "%m-%d %H:%M:%S") -> str:
        if value is None or isinstance(value, bool):
            return "N/A"
        if isinstance(value, (int, float)):
            vf = normalize_epoch_sec(value, default=None)
            if vf is None:
                return str(value)
            try:
                return datetime.fromtimestamp(vf).strftime(fmt)
            except Exception:
                return str(value)
        if hasattr(value, "strftime"):
            try:
                return value.strftime(fmt)
            except Exception:
                return str(value)
        return str(value)

    @staticmethod
    def is_new_coin_alert(alert: Any) -> bool:
        return isinstance(alert, dict) and alert.get("type") == "new_coin_discovered"

    @staticmethod
    def utf8_header(value: Any, default: str = "") -> str:
        """Encode an HTTP header while preserving non-ASCII symbols.

        Headers are Latin-1, while ntfy decodes this value as UTF-8, so pass UTF-8 bytes
        through Latin-1 to preserve the original symbol.
        """
        cleaned = sanitize_single_line(value, default)
        return cleaned.encode("utf-8").decode("latin-1", "ignore") if cleaned else ""

    @staticmethod
    def _extract_price_alert_fields(alert: Any) -> tuple[str, str, float, float, float, str, Any]:
        ref_time_val = obj_attr(alert, "reference_time") or obj_attr(alert, "timestamp")
        return (
            obj_attr(alert, "symbol", "N/A"),
            obj_attr(alert, "alert_type", ""),
            safe_float(obj_attr(alert, "percent_change", 0.0), 0.0),
            safe_float(obj_attr(alert, "current_price", 0.0), 0.0),
            safe_float(obj_attr(alert, "reference_price", 0.0), 0.0),
            AlertNotifier.format_human_readable_time(ref_time_val),
            obj_attr(alert, "url"),
        )

    @staticmethod
    def format_new_coin_message(alert: dict) -> str:
        raw_price = alert.get("price")
        px_val = safe_pos_float(raw_price, default=None)
        if px_val is not None:
            price_str = f"Price: ${px_val:.4f}"
        elif clean_scalar_str(raw_price) is not None and safe_float(raw_price, default=None) is None:
            price_str = f"Price: {raw_price}"
        else:
            price_str = "Price: N/A"
        lines = [
            f"\U0001f195: {alert.get('symbol', 'N/A')} - {alert.get('name', alert.get('symbol', 'N/A'))}",
            f"Source: {alert.get('source', 'unknown')}",
            f"Added: {AlertNotifier.format_human_readable_time(alert.get('added_at'))}",
            price_str,
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
        body = str(alert.get("body") or "").strip()
        head = body or str(alert.get("name", alert.get("symbol", "?")))
        parts = [head, str(alert.get("source", "?"))]
        ts = alert.get("added_at")
        if ts is not None and not isinstance(ts, bool) and (hasattr(ts, "strftime") or isinstance(ts, (int, float))):
            formatted = AlertNotifier.format_human_readable_time(ts, fmt="%m-%d %H:%M")
            if formatted != str(ts):
                parts.append(formatted)
        return " \u00b7 ".join(parts)

    @staticmethod
    def format_batch_message(alerts) -> str:
        lines = []
        for alert in alerts:
            if isinstance(alert, dict) and alert.get("type") == "bot_event":
                lines.append(AlertNotifier.format_bot_event(alert))
                continue
            if AlertNotifier.is_new_coin_alert(alert):
                lines.append(AlertNotifier.format_new_coin_message(alert))
                continue

            symbol, alert_type, pct_change, current_price, ref_price, reference_time, url = (
                AlertNotifier._extract_price_alert_fields(alert)
            )
            direction = "U" if alert_type == "up" else "D"
            esc_tier = exact_pos_int(obj_attr(alert, "escalation_tier", 1), 1)
            prev_pct = safe_float(obj_attr(alert, "previous_percent_change", None), default=None)
            esc_tag = ""
            if esc_tier > 1:
                prev_part = f", prev {prev_pct:+.2f}%" if prev_pct is not None else ""
                esc_tag = f" [ESC #{esc_tier}{prev_part}]"

            lines.append(
                f"{symbol}: {direction} {pct_change:+.2f}%{esc_tag} "
                f"| C ${current_price:.4f} | R ${ref_price:.4f} "
                f"({reference_time})"
            )

            if url:
                lines.append(f"Link: {url}")

        return "\n".join(lines)

    @staticmethod
    def save_to_file(alert, filename="logs/alerts.log"):
        if (
            filename is None
            or isinstance(filename, bool)
            or not isinstance(filename, (str, os.PathLike))
            or not str(filename).strip()
        ):
            filename = "logs/alerts.log"
        alert_file = Path(str(filename).strip())
        if not alert_file.is_absolute():
            alert_file = BASE_DIR / alert_file
        try:
            alert_file.parent.mkdir(parents=True, exist_ok=True)
            now_iso = datetime.now().isoformat()
            with alert_file.open("a", encoding="utf-8") as f:
                if AlertNotifier.is_new_coin_alert(alert):
                    f.write(
                        f"[{now_iso}] NEW COIN {alert.get('symbol')} "
                        f"(source: {alert.get('source', 'unknown')})\n"
                        f"{AlertNotifier.format_new_coin_message(alert)}\n"
                    )
                elif isinstance(alert, dict):
                    if alert.get("type") == "bot_event":
                        f.write(
                            f"[{now_iso}] BOT EVENT {alert.get('name', alert.get('symbol', 'N/A'))}\n"
                            f"{AlertNotifier.format_bot_event(alert)}\n"
                        )
                    else:
                        body = alert.get("body") or alert.get("name") or str(alert)
                        f.write(f"[{now_iso}] {alert.get('symbol', 'N/A')}: {body}\n")
                else:
                    sym, atype, pchg, cprice, rprice, reference_time, url = (
                        AlertNotifier._extract_price_alert_fields(alert)
                    )
                    f.write(
                        f"[{now_iso}] {sym} - {atype} - {pchg:+.2f}%\n"
                        f"  Price: ${cprice:.4f}\n"
                        f"  Reference: ${rprice:.4f} (at {reference_time})\n"
                    )
                    if url:
                        f.write(f"  Link: {url}\n")
                f.write("-" * 50 + "\n")
            return True
        except Exception as e:
            print(f"[Notifier] File exception: {e}")
            return False

    @staticmethod
    def _collect_batch_symbols_and_items(alerts: list) -> tuple[list[str], list[str]]:
        symbols: list[str] = []
        items: list[str] = []
        for a in alerts:
            sym = obj_attr(a, "symbol", "N/A")
            if sym and sym != "N/A":
                if sym not in symbols:
                    symbols.append(sym)
                atype = obj_attr(a, "alert_type", "")
                dir_str = "\u25b2" if atype == "up" else ("\u25bc" if atype == "down" else "")
                items.append(f"{sym} {dir_str}".strip())
        return symbols, items

    @staticmethod
    def _format_price_alerts_batch_title(alerts: list) -> str:
        count = len(alerts)
        symbols, items = AlertNotifier._collect_batch_symbols_and_items(alerts)
        if len(symbols) == 1:
            return f"{symbols[0]} ({count} alerts)"
        for candidate in (", ".join(items), ", ".join(symbols)):
            if candidate and len(candidate) <= 50:
                return f"Price Alerts ({count}): {candidate}"
        return f"Price Alerts ({count} coins)"

    @staticmethod
    def _format_new_coins_batch_title(alerts: list) -> str:
        count = len(alerts)
        symbols, _ = AlertNotifier._collect_batch_symbols_and_items(alerts)
        syms_str = ", ".join(symbols)
        if syms_str and len(syms_str) <= 50:
            return f"New Coins ({count}): {syms_str}"
        return f"New Coins Discovered ({count})"

    @staticmethod
    def email_subject(alerts: list) -> str:
        if not alerts:
            return "Alert"
        first = alerts[0]
        if isinstance(first, dict) and first.get("type") == "new_coin_discovered":
            return (
                AlertNotifier._format_new_coins_batch_title(alerts)
                if len(alerts) > 1
                else f"New Coin: {first.get('symbol', 'N/A')}"
            )
        if (isinstance(first, dict) and "alert_type" in first) or hasattr(first, "alert_type"):
            return AlertNotifier._format_price_alerts_batch_title(alerts)
        if isinstance(first, dict):
            return str(first.get("name") or first.get("title") or "Alert")
        return "Alert"

    @staticmethod
    def send_email_batch(
        alerts: list[dict],
        recipient: Optional[str] = None,
        subject: Optional[str] = None,
    ) -> bool:
        if not alerts or _external_notifications_disabled():
            return bool(alerts)
        alerts = list(alerts)
        subject = subject or AlertNotifier.email_subject(alerts)
        message = AlertNotifier.format_batch_message(alerts)
        if _is_orchestrated():
            # The orchestrator parses child stdout; same schema as its retry queue.
            return _emit_orchestrator_intent(
                {"__orchestrator_intent__": "email", "subject": subject, "message": message},
                default=str,
            )
        return _get_notification_dispatcher()._send_email(subject, message)

    @staticmethod
    def send_phone_webhook_batch(alerts, webhook_url: Optional[str] = None):
        if not alerts or _external_notifications_disabled():
            return bool(alerts)
        if _is_orchestrated():
            return _emit_orchestrator_intent(
                {
                    "__orchestrator_intent__": "ntfy_webhook",
                    "webhook_url": webhook_url,
                    "alerts": [
                        {**{"type": "price_alert"}, **a.to_dict()}
                        if hasattr(a, "to_dict")
                        else (a if isinstance(a, dict) else str(a))
                        for a in alerts
                    ],
                },
                default=lambda obj: (
                    obj.to_dict()
                    if hasattr(obj, "to_dict")
                    else (obj.isoformat() if isinstance(obj, datetime) else str(obj))
                ),
            )
        return _get_notification_dispatcher().dispatch_alerts(list(alerts), webhook_url=webhook_url)

    @staticmethod
    def send(alert, enable_console=True, enable_file=True,
             enable_email=False, enable_phone_webhook=False, webhook_url=None,
             filename="logs/alerts.log"):
        alerts = [alert] if not isinstance(alert, list) else alert
        if _parse_bool_flag(enable_console, True):
            for item in alerts:
                print(f"\n{'=' * 70}\n{item}\n{'=' * 70}")
        if _parse_bool_flag(enable_file, True):
            for item in alerts:
                AlertNotifier.save_to_file(item, filename=filename)
        if _parse_bool_flag(enable_email, False):
            AlertNotifier.send_email_batch(alerts)
        if _parse_bool_flag(enable_phone_webhook, False):
            AlertNotifier.send_phone_webhook_batch(alerts, webhook_url=webhook_url)


def _is_orchestrated() -> bool:
    """Return True when running as a managed child process emitting JSON intents to stdout."""
    return bool_env("MPTRADE_ORCHESTRATED", default=False)


def _external_notifications_disabled() -> bool:
    """Return True when external notifications (ntfy, email, orchestrator intents) are suppressed."""
    return is_replay_or_backtest_env("DISABLE_EXTERNAL_NOTIFICATIONS")


_GUARD_MARKERS = (
    "\U0001f6d1", "\U0001f6e1", "STOP-LOSS", "STOP_LOSS", "SMART-SL", "POST-STOP", "TRAILING", "LIQUID", "CATASTROPH", "CRASH",
    "LICHID", "CATASTROF", "IMBALANCE", "DEZECHILIBR", "QUARANTINE", "SL ", " SL",
    "REFUSED", "BLOCKED", "SKIPPED",
)
_PRIMARY_ERROR_MARKERS = (
    "ERROR", "FAILED", "EXCEPTION", "CRITICAL", "GIVE UP", "CORRUPTION", "REJECTED", "\u2717", "UNRECOGNIZED",
)
_BILINGUAL_ERROR_MARKERS = ("ESUAT", "ERORI", "DISPARUT", "GONE")
_OPS_MARKERS = (*_PRIMARY_ERROR_MARKERS, *_BILINGUAL_ERROR_MARKERS, "MANUAL")
_EXPLICIT_ERROR_MARKERS = (*_PRIMARY_ERROR_MARKERS, *_BILINGUAL_ERROR_MARKERS, "CRASH")
_SERVER_MARKERS = (
    "CONFIG RELOAD", "CONFIG CHANGED", "CONFIG PROPOSAL", "PILOT BACKTEST", "BACKTEST:",
    "SERVER STATUS", "AUTODEPLOY",
    "BOT RESTARTED", "BOT AUTO-STARTED", "FLEET RESTARTED", "SERVICE RESTARTED",
    "PROCESS RESTARTED", "DAEMON RESTARTED", "DNS RESOLVER RESTARTED",
    "ZOMBIE BOT TERMINATED", "PROCESS RESOURCE USAGE RECOVERED",
)
_DEADMAN_MARKERS = ("SERVER DOWN", "DEADMAN")
_NON_LIVE_SELL_MARKERS = ("EXPIRED", "CANCEL", "SHADOW", "REFUSED", "BLOCKED", "SKIPPED", "WOULD ")
_WORD_SELL_RE = re.compile(r"\bSELL\b", re.IGNORECASE)
_PRICE_ALERT_SOURCES = ("coinmarketcap", "coingecko", "dexscreener", "pricechecker", "notifier")
_NOTIFICATION_DISPATCHER_CLS: Any = None


def register_notification_dispatcher_cls(cls: Any) -> None:
    global _NOTIFICATION_DISPATCHER_CLS
    _NOTIFICATION_DISPATCHER_CLS = cls


def _get_notification_dispatcher() -> Any:
    global _NOTIFICATION_DISPATCHER_CLS
    disp_mod = sys.modules.get("notify_engine.ntfydispatcher")
    if disp_mod is not None and hasattr(disp_mod, "NtfyDispatcher"):
        return disp_mod.NtfyDispatcher()
    if _NOTIFICATION_DISPATCHER_CLS is None:
        disp_mod = load_local_module(
            os.path.dirname(os.path.abspath(__file__)),
            "ntfydispatcher",
            module_key="notify_engine.ntfydispatcher",
        )
        _NOTIFICATION_DISPATCHER_CLS = getattr(disp_mod, "NtfyDispatcher", None)
    return _NOTIFICATION_DISPATCHER_CLS()


def _clean_str(val: Any, default: str = "", *, strip: bool = False) -> str:
    if isinstance(val, bool) or val is None:
        return default
    s = str(val).strip() if strip else str(val)
    return s if s else default


def resolve_topic(category: str) -> str:
    cat = _clean_str(category, strip=True).upper()
    if cat == "DEADMAN":
        if not optional_env("NTFY_TOPIC_DEADMAN") and not optional_env("NTFY_TOPIC_ERROR"):
            load_env_stack()
        return optional_env("NTFY_TOPIC_DEADMAN") or required_env("NTFY_TOPIC_ERROR")
    if cat in ("SERVER", "SERVER_STATE", "SERVER_STATUS"):
        cat = "SERVER"
    key = f"NTFY_TOPIC_{cat}"
    if not optional_env(key):
        load_env_stack()
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
    if any(m in t_no_owner for m in _DEADMAN_MARKERS) or "deadman" in s:
        return "DEADMAN"
    if any(m in t_no_owner for m in _PRIMARY_ERROR_MARKERS):
        return "ERROR"
    if any(m in t_no_owner for m in _SERVER_MARKERS) or "scheduled_pilot" in s:
        return "SERVER"
    if "watchdog" in s:
        return "ERROR"
    if any(m in t_no_owner for m in _GUARD_MARKERS) or any(m in s for m in ("guard", "trail", "assetguardian", "stop")):
        return "GUARD"

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
        or any(m in t for m in ("\u25b2", "\u25bc", "\U0001f4c8", "\U0001f4c9", "RISE", "DROP"))
    ):
        return "PRICE"
    return "TRADES"


def _topic_for_category(title: str, source: str) -> str:
    cat = _category_for_title_and_source(title, source)
    return resolve_topic(cat)


EMAIL_MIRRORED_CATEGORIES = ("ERROR", "DEADMAN")

_EMAIL_SUPPRESSED_SUBJECT_MARKERS = (
    "DEV CI REGRESSION",
    "DEV CLI REGRESSION",
    "BOT RESTARTED",
    "BOT RESTART",
    "BOT AUTO-STARTED",
    "FLEET RESTARTED",
    "SERVICE RESTARTED",
    "PROCESS RESTARTED",
    "DAEMON RESTARTED",
)

_VPN_RECOVERED_SUBJECT_MARKERS = (
    "PIA VPN: RECOVERED",
    "PIA VPN: RESTORED TO PRIMARY DIP",
    "PIA VPN: RECONNECTED AFTER COOLDOWN",
)

_VPN_STATIC_IP_BODY_MARKERS = (
    "DEDICATED-",
    "DEDICATED IP",
)


def is_email_suppressed_subject(subject: str | None, body: str | None = None) -> bool:
    """Return True for CI regression, routine bot/fleet restarts, or static-IP VPN recoveries (push-only)."""
    if not subject or not isinstance(subject, str):
        return False
    s = subject.strip().upper()
    if not s:
        return False
    if any(m in s for m in _EMAIL_SUPPRESSED_SUBJECT_MARKERS):
        return True
    if any(m in s for m in _VPN_RECOVERED_SUBJECT_MARKERS):
        b = str(body or "").strip().upper()
        if b and any(m in b for m in _VPN_STATIC_IP_BODY_MARKERS):
            return True
    return False


def is_email_mirrored(topic: str | None) -> bool:
    if not topic or not isinstance(topic, str) or not topic.strip():
        return False
    if not optional_env("NTFY_TOPIC_ERROR"):
        load_env_stack()
    topics = {
        optional_env(f"NTFY_TOPIC_{cat}")
        for cat in EMAIL_MIRRORED_CATEGORIES
    }
    topics.discard("")
    if not topics:
        topics.update({"ntfy-error-1978"})
    return _canonical_topic_key(topic) in topics


_XSTOCK_TITLE_TOKEN_RE = re.compile(r"\b[A-Z][A-Z0-9]{0,9}x(?:USD|EUR|USDC|USDT|\.T)\b")


def should_email_trade_sell_alert(title: str, symbol: str = "", source: str = "") -> bool:
    """Return True when a live SELL trade notification should also be mirrored to email.

    Matches either any tokenized xStock SELL (regardless of ticker/coin) when
    ``NOTIFICATION_EMAIL_SELL_XSTOCKS`` is enabled, or any live SELL whose canonical
    base asset appears in ``NOTIFICATION_EMAIL_SELL_BASES``.
    """
    raw_bases = optional_env("NOTIFICATION_EMAIL_SELL_BASES")
    raw_xstocks = optional_env("NOTIFICATION_EMAIL_SELL_XSTOCKS")
    if not raw_bases and not raw_xstocks and not optional_env("NTFY_TOPIC_ERROR"):
        load_env_stack()
        raw_bases = optional_env("NOTIFICATION_EMAIL_SELL_BASES")
        raw_xstocks = optional_env("NOTIFICATION_EMAIL_SELL_XSTOCKS")
    bases = {
        canonical_base_asset(b) or str(b).strip().upper()
        for b in split_csv(raw_bases or "")
        if str(b).strip()
    }
    bases.discard("")
    email_xstocks = _parse_bool_flag(raw_xstocks, False)
    if not bases and not email_xstocks:
        return False
    raw_title_no_owner = strip_owner_tags(_clean_str(title))
    t_clean = raw_title_no_owner.upper()
    if not t_clean or not _WORD_SELL_RE.search(t_clean):
        return False
    if any(m in t_clean for m in _NON_LIVE_SELL_MARKERS):
        return False
    if _category_for_title_and_source(title, source) != "TRADES":
        return False
    sym_s = _clean_str(symbol, strip=True)
    if email_xstocks:
        if sym_s and is_kraken_tokenized_asset(sym_s):
            return True
        if _XSTOCK_TITLE_TOKEN_RE.search(raw_title_no_owner):
            return True
    if sym_s and bases:
        sym_base = canonical_base_asset(sym_s) or sym_s.upper()
        if sym_base in bases:
            return True
    return any(re.search(rf"\b{re.escape(b)}(?:X?USD[CT]?|_US_EQ)?\b", t_clean) for b in bases)


def notify(title: str, body: str, source: str, symbol: str,
           price: float = None, desktop: bool = False,
           email: bool = None) -> None:
    """Shared bot-event notifier: ntfy (routed by category) plus optional email.

    ERROR-category events reach email automatically through the dispatcher's topic mirror
    (notify_engine.ntfymailer). ``email=True`` or a SELL matching ``NOTIFICATION_EMAIL_SELL_XSTOCKS``
    / ``NOTIFICATION_EMAIL_SELL_BASES`` forces a copy for any other category.
    """
    if _external_notifications_disabled():
        return
    if is_fake_alert_text(title, body, symbol, source):
        return
    alert = make_bot_event_alert(title, body, source, symbol)
    if _is_orchestrated():
        intent: dict[str, Any] = {"__orchestrator_intent__": "ntfy_webhook", "alerts": [alert]}
        if email is False:
            intent["skip_email"] = True
        if _AUTO_TOPIC_TAG_RE.search(str(title or "")):
            urgent = _alerts_are_urgent([alert])
            intent.update({
                "title": title,
                "message": body,
                "priority": "urgent" if urgent else "high",
                "topic": _topic_for_category(title, source),
            })
        _emit_orchestrator_intent(intent)
    else:
        _get_notification_dispatcher().dispatch_alerts([alert], skip_email=(email is False))
    force_email = _parse_bool_flag(email, False) or (
        email is not False and should_email_trade_sell_alert(title, symbol, source)
    )
    if force_email and not is_email_suppressed_subject(title, body):
        if not is_email_mirrored(_topic_for_category(title, source)):
            AlertNotifier.send_email_batch([alert], subject=title)


def bind_notify(symbol_env_keys: tuple, default_symbol: str):
    def bound_notify(title: str, body: str, source: str, price: float = None, desktop: bool = False, symbol: str = None, email: bool = None):
        sym_clean = symbol.strip() if isinstance(symbol, str) and symbol.strip() else None
        resolved = sym_clean or next(
            (val for key in symbol_env_keys if (val := optional_env(key))),
            default_symbol,
        )
        notify(title, body, source, resolved, price=price, desktop=desktop, email=email)
    bound_notify.__name__ = "notify"
    return bound_notify


def dispatch_order_event_notification(
    symbol: Any,
    side: Any,
    order_id: Any,
    price: Any,
    quantity: Any,
    *,
    default_verb: str,
    default_icon: str,
    venue: Optional[str] = None,
    owner: Optional[str] = None,
    source: Optional[str] = None,
    order_type: Optional[str] = None,
    allow_ai_executor: bool = True,
    trade_tag: Optional[str] = None,
    notional_usd: Optional[float] = None,
    thesis: Optional[str] = None,
    title: Optional[str] = None,
    body: Optional[str] = None,
    notify_fn: Optional[Any] = None,
    include_price: bool = True,
) -> bool:
    """Dispatch a standardized order placement or execution notification."""
    sym_s = str(symbol or "TRADE").strip() or "TRADE"
    owner_norm = str(owner or "").strip().lower()
    if owner_norm == "ai_executor" and not _parse_bool_flag(allow_ai_executor, True):
        return False
    px_f = safe_pos_float(price, default=0.0) or 0.0
    qty_f = safe_pos_float(quantity, default=0.0) or 0.0
    notional_f = safe_pos_float(notional_usd, default=None)
    if notional_f is None:
        notional_f = px_f * qty_f
    side_s = str(side or "").strip().upper() or "ORDER"
    ord_type_s = str(order_type or "MARKET").strip().upper()
    venue_s = str(venue or source or "binance").strip()
    tag_s = str(trade_tag).strip() if trade_tag and not isinstance(trade_tag, bool) else ""
    eff_owner = tag_s or owner_norm or None
    verb_raw = str(default_verb or "FILLED").strip().upper()
    if verb_raw == "EXECUTED":
        status_verb = "FILLED"
    elif verb_raw.startswith("PLACED"):
        status_verb = "PLACED"
    else:
        status_verb = verb_raw or "FILLED"
    gen_title, gen_body = format_trade_notification(
        status_verb,
        side_s,
        sym_s,
        px_f,
        qty_f,
        notional_f,
        None,
        order_id,
        venue=venue_s,
        owner=eff_owner,
        detail=ord_type_s,
        thesis=thesis if (thesis is not None and not isinstance(thesis, bool)) else None,
    )
    eff_title = title or gen_title
    eff_body = body or gen_body
    if callable(notify_fn):
        if include_price:
            notify_fn(title=eff_title, body=eff_body, symbol=sym_s, price=px_f)
        else:
            notify_fn(title=eff_title, body=eff_body, symbol=sym_s)
        return True
    notify(
        title=eff_title,
        body=eff_body,
        source=str(source or venue or owner or "llm_trader"),
        symbol=sym_s,
        price=px_f if include_price else None,
    )
    return True
