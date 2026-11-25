#!/usr/bin/env python3
"""portfolio_snapshot.py -- UNIFIED cross-engine view (Kraken base v2 + HL long-hold).

Read-only monitoring: reads the bot states plus the current price and reports
positions, realised + unrealised P&L, exposure and drawdown per engine and in
total. It touches nothing.

  ./myenv/bin/python verify_tools/portfolio_snapshot.py            # one shot (table)
  ./myenv/bin/python verify_tools/portfolio_snapshot.py --json     # one JSON line (for cron/logs)
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KRAKEN_DIR = os.path.join(ROOT, "kraken")
CACHEDB_DIR = os.path.join(ROOT, "cachedb")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")
with contextlib.redirect_stdout(sys.stderr):
    from botcore import (  # noqa: E402
        _parse_bool_flag,
        add_store_true_cli_args,
        http_request,
        is_live_network_backfill_enabled,
        optional_env,
        parse_dotenv,
        parse_iso_datetime_utc,
        required_float_env,
        safe_float,
        safe_nonneg_float,
        safe_pos_float,
    )
    from intelligence.macro_sentiment_ssot import load_venue_balances_state  # noqa: E402
    from instrument_registry import select_instruments  # noqa: E402
    from notify_engine import alertnotifiers as _alertnotifiers  # noqa: E402
    from providers.market_api import api as mkt  # noqa: E402
    from providers.portfolio_state_store import (  # noqa: E402
        load_venue_balance_caches,
        resolve_kraken_trade_cache_position,
    )
    from state_io import encode_json_payload, read_json_dict, save_json_state  # noqa: E402
from strategies.state_store import state_path_for  # noqa: E402

_ROOT_CONFIG = parse_dotenv(os.path.join(ROOT, "config.env"))


def _policy_float(key: str) -> float:
    value = optional_env(key) or _ROOT_CONFIG.get(key)
    return required_float_env(key, {key: value})

# Monitored Kraken pairs derived dynamically from instruments.conf SSOT (enabled Kraken instruments).
KRAKEN_PAIRS = [
    (spec.symbol, spec.base, False)
    for spec in select_instruments("kraken", only_enabled=True).values()
] or [("HYPEUSD", "HYPE", False)]


def _load_state(pair: str) -> dict | None:
    path = state_path_for(pair, KRAKEN_DIR)
    return read_json_dict(path, default=None, recover_previous=True)


def _kraken_price(pair: str):
    try:
        return mkt.get_current_price(pair, provider_name="kraken")
    except Exception:  # noqa: BLE001 -- the snapshot must not die on one missed price
        return None


def _live_kraken_held_qty(
    vb_state: dict,
    pair: str,
    base_label: str,
    *,
    kraken_bal_cache: dict | None = None,
) -> float | None:
    if isinstance(vb_state, dict) and vb_state:
        pos_entry = vb_state.get(f"pos:kraken:{pair}")
        if isinstance(pos_entry, dict):
            free_q = safe_nonneg_float(pos_entry.get("free_base"), default=None)
            locked_q = safe_nonneg_float(pos_entry.get("locked_base_qty"), default=0.0) or 0.0
            if free_q is not None:
                return free_q + locked_q
    if isinstance(kraken_bal_cache, dict) and kraken_bal_cache:
        for key in (base_label, base_label.upper()):
            entry = kraken_bal_cache.get(key)
            if isinstance(entry, dict):
                free_q = safe_nonneg_float(entry.get("free"), default=None)
                locked_q = safe_nonneg_float(entry.get("locked"), default=0.0) or 0.0
                if free_q is not None:
                    return free_q + locked_q
    if isinstance(vb_state, dict) and vb_state:
        for key in (f"kraken:{base_label}", f"kraken:{base_label.upper()}"):
            val = vb_state.get(key)
            if not isinstance(val, (dict, list, bool)):
                if (q := safe_nonneg_float(val, default=None)) is not None:
                    return q
    return None


def kraken_rows() -> list[dict]:
    rows = []
    _, vb_state, _ = load_venue_balances_state(CACHEDB_DIR)
    bal_caches = load_venue_balance_caches(CACHEDB_DIR)
    kraken_bal_items = (
        bal_caches.get("kraken", {}).get("items")
        if isinstance(bal_caches, dict) and isinstance(bal_caches.get("kraken"), dict)
        else None
    )
    for pair, label, paper in KRAKEN_PAIRS:
        live_qty = _live_kraken_held_qty(vb_state, pair, label, kraken_bal_cache=kraken_bal_items)
        cache_pos = resolve_kraken_trade_cache_position(pair, CACHEDB_DIR, held_qty=live_qty)
        s = _load_state(pair)
        if cache_pos is None and s is None:
            continue
        if isinstance(cache_pos, dict):
            qty = safe_float(cache_pos.get("qty"), 0.0)
            cost = safe_float(cache_pos.get("cost"), 0.0)
            realized_net = safe_float(cache_pos.get("realized_net"), 0.0)
            spent = safe_float(cache_pos.get("spent"), 0.0)
        else:
            qty = safe_float(s.get("qty"), 0.0)
            cost = safe_float(s.get("cost"), 0.0)
            realized_net = safe_float(s.get("realized_net"), 0.0)
            spent = safe_float(s.get("spent"), 0.0)
        px_opt = safe_pos_float(_kraken_price(pair), default=None)
        px = px_opt if px_opt is not None else 0.0
        upnl = (px - cost / qty) * qty if qty > 1e-12 and cost > 0.0 and px > 0 else 0.0
        # Unrealised drawdown as a % of cost (this triggers the risk alert).
        dd_pct = (upnl / cost * 100.0) if cost > 1e-9 else 0.0
        raw_orders = s.get("orders") if isinstance(s, dict) else None
        open_orders_cnt = len(raw_orders) if isinstance(raw_orders, (list, tuple, dict)) else 0
        rows.append({
            "engine": f"Kraken {label}" + (" (paper)" if paper else ""),
            "paper": paper,
            "qty": round(qty, 6),
            "value_usd": round(qty * px, 2),
            "cost": round(cost, 2),
            "dd_pct": round(dd_pct, 2),
            "price_ok": px_opt is not None,
            "realized_net": round(realized_net, 2),
            "unrealized": round(upnl, 2),
            "spent": round(spent, 0),
            "cycle": s.get("cycle") if isinstance(s, dict) else None,
            "dca_buys": s.get("dca_buys") if isinstance(s, dict) else None,
            "open_orders": open_orders_cnt,
        })
    return rows


def hl_row() -> dict | None:
    try:
        hl_prov = mkt.provider_by_name("hyperliquid")
        if hl_prov is None:
            raise RuntimeError("hyperliquid provider unavailable")
        summary = hl_prov.spot_balances_and_mid("HYPE", "USDC")
        hype = safe_float(summary.get("qty"), 0.0)
        usdc = safe_float(summary.get("cash_usd"), 0.0)
        mid = safe_float(summary.get("mid"), 0.0)
        return {"engine": "HL HYPE (long-hold)", "paper": False, "qty": round(hype, 4),
                "value_usd": round(hype * mid, 2), "cash_usd": round(usdc, 2)}
    except Exception as e:  # noqa: BLE001
        return {"engine": "HL", "error": str(e)[:60]}


ALERT_STATE = os.path.join(ROOT, "logs", ".risk_alert_state.json")


def _ntfy(title: str, body: str) -> None:
    """Send an ntfy notification (best-effort, no external dependencies)."""
    if _alertnotifiers._external_notifications_disabled():
        return
    topic = optional_env("NTFY_TOPIC") or optional_env("NTFY_TOPIC_GUARD")
    if not topic:
        for cfg in (os.path.join(ROOT, ".env"), os.path.join(KRAKEN_DIR, "config.env")):
            if os.path.exists(cfg):
                try:
                    env_map = parse_dotenv(cfg)
                    topic = (env_map.get("NTFY_TOPIC") or env_map.get("NTFY_TOPIC_GUARD") or "").strip()
                    if topic:
                        break
                except OSError:
                    topic = ""
    url = _alertnotifiers.format_ntfy_url(topic)
    if not url:
        return
    clean_title = _alertnotifiers.sanitize_single_line(title, "Alert")
    body_str = "" if isinstance(body, bool) else str(body or "")
    if is_live_network_backfill_enabled():
        allowed, _, _, out_payload = _alertnotifiers.pass_common_gate(
            "ntfy", clean_title, body_str, "high", _alertnotifiers._canonical_topic_key(url),
        )
        if not allowed:
            return
        if out_payload:
            clean_title, body_str, _, out_topic = out_payload
            url = _alertnotifiers.format_ntfy_url(out_topic or topic) or url
    safe_title = _alertnotifiers.AlertNotifier.utf8_header(clean_title) or "Alert"
    try:
        http_request(
            "POST",
            url,
            headers={"Title": safe_title, "Priority": "high", "Tags": "warning"},
            body=body_str.encode("utf-8"),
            timeout=10.0,
        )
    except Exception:  # noqa: BLE001 -- best-effort alert, we do not block the snapshot
        pass


def run_alerts(kr: list[dict]) -> list[str]:
    """Check the drawdown of every REAL position and send an ntfy alert past the threshold.

    Threshold and re-alert policy are mandatory values from root config.env.
    """
    threshold = abs(_policy_float("RISK_DD_ALERT_PCT"))
    realert_hours = abs(_policy_float("RISK_DD_REALERT_HOURS"))
    worsen_pct = abs(_policy_float("RISK_DD_WORSEN_PCT"))
    now = datetime.now(timezone.utc)
    st = read_json_dict(ALERT_STATE, default={}, recover_previous=True)
    fired = []
    for r in kr:
        if not isinstance(r, dict):
            continue
        engine = str(r.get("engine") or "").strip()
        if not engine or _parse_bool_flag(r.get("paper"), False) or safe_float(r.get("qty"), 0.0) <= 1e-9:
            continue
        if r.get("price_ok") is False:
            continue
        dd = safe_float(r.get("dd_pct"), 0.0)  # Negative means a loss.
        if dd > -threshold:
            st.pop(engine, None)  # Back under the threshold -> reset.
            continue
        prev = st.get(engine, {})
        prev = prev if isinstance(prev, dict) else {}
        prev_dd = safe_float(prev.get("dd"), 0.0)
        prev_ts = prev.get("ts")
        stale = True
        if prev_ts and not isinstance(prev_ts, bool) and (parsed_ts := parse_iso_datetime_utc(str(prev_ts))) is not None:
            elapsed_s = (now - parsed_ts).total_seconds()
            stale = elapsed_s < 0 or elapsed_s > realert_hours * 3600
        if stale or dd <= prev_dd - worsen_pct:
            unreal = safe_float(r.get("unrealized"), 0.0)
            cost = safe_float(r.get("cost"), 0.0)
            _ntfy(
                f"⚠️ Risk {engine}: DD {dd:.1f}%",
                f"{engine} unrealized drawdown {dd:.1f}% "
                f"(unrealized ${unreal:.0f} on cost ${cost:.0f}, "
                f"threshold {threshold:.0f}%). Check the position.")
            st[engine] = {"dd": dd, "ts": now.isoformat(timespec="seconds")}
            fired.append(f"{engine} DD {dd:.1f}%")
    try:
        save_json_state(ALERT_STATE, st)
    except (OSError, ValueError, TypeError):
        pass
    return fired


def main() -> int:
    ap = argparse.ArgumentParser(description="Cross-engine portfolio snapshot (read-only).")
    add_store_true_cli_args(
        ap,
        ("--json", "one JSON line (for cron/logs)"),
        ("--alert", "check per-position drawdown and send an ntfy alert past the threshold"),
    )
    args = ap.parse_args()

    stdout_guard = contextlib.redirect_stdout(sys.stderr) if args.json else contextlib.nullcontext()
    with stdout_guard:
        kr = kraken_rows()
        hl = hl_row()
        real_total = sum(safe_float(r.get("realized_net"), 0.0) for r in kr if isinstance(r, dict) and not _parse_bool_flag(r.get("paper"), False))
        upnl_total = sum(safe_float(r.get("unrealized"), 0.0) for r in kr if isinstance(r, dict) and not _parse_bool_flag(r.get("paper"), False))
        snap = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "kraken": kr, "hl": hl,
                "kraken_real_realized_net": round(real_total, 2),
                "kraken_real_unrealized": round(upnl_total, 2)}

        if args.alert:
            fired = run_alerts(kr)
            snap["alerts_fired"] = fired

    if args.json:
        # Write raw JSONL directly to stdout so log.py's timestamped builtins.print never corrupts JSONL records.
        sys.stdout.write(encode_json_payload(snap) + "\n")
        sys.stdout.flush()
        return 0

    if args.alert:
        print(f"  risk-alert: {'; '.join(fired) if fired else 'nothing over the threshold'}")

    print(f"=== PORTFOLIO {snap['ts']} ===")
    print(f"  {'engine':<22}{'qty':>12}{'value$':>11}{'realised$':>11}{'unreal$':>10}{'budget':>10}")
    for r in kr:
        print(f"  {r['engine']:<22}{r['qty']:>12.4f}{r['value_usd']:>11.2f}"
              f"{r['realized_net']:>11.2f}{r['unrealized']:>10.2f}{str(r['spent'])+'':>10}")
    if hl and "error" not in hl:
        print(f"  {hl['engine']:<22}{hl['qty']:>12.4f}{hl['value_usd']:>11.2f}"
              f"{'—':>11}{'—':>10}  cash ${hl.get('cash_usd', 0):.0f}")
    elif hl:
        print(f"  HL: error ({hl['error']})")
    print(f"  --- Kraken REAL: realized ${real_total:+.2f} | unrealized ${upnl_total:+.2f} ---")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
