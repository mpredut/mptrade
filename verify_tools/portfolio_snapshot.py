#!/usr/bin/env python3
"""portfolio_snapshot.py — UNIFIED cross-engine view (Kraken base v2 + HL long-hold).

Read-only monitoring: reads the bot states plus the current price and reports
positions, realised + unrealised P&L, exposure and drawdown per engine and in
total. It touches nothing.

  ./myenv/bin/python verify_tools/portfolio_snapshot.py            # one shot (table)
  ./myenv/bin/python verify_tools/portfolio_snapshot.py --json     # one JSON line (for cron/logs)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KRAKEN_DIR = os.path.join(ROOT, "kraken")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from botcore import _parse_bool_flag, parse_dotenv, required_float_env, safe_float  # noqa: E402
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")
from state_io import read_recovering_previous, save_json_state  # noqa: E402

_ROOT_CONFIG = parse_dotenv(os.path.join(ROOT, "config.env"))


def _policy_float(key: str) -> float:
    value = os.environ.get(key, _ROOT_CONFIG.get(key))
    return required_float_env(key, {key: value})

# Monitored Kraken pairs (pair -> label, is_paper).
KRAKEN_PAIRS = [("HYPEUSD", "HYPE", False), ("TAOUSD", "TAO", False), ("ADAUSD", "ADA", True)]


def _load_state(pair: str) -> dict | None:
    path = os.path.join(KRAKEN_DIR, f".state_{pair}.json")
    try:
        data = read_recovering_previous(path, json.loads, default=None)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _kraken_price(pair: str):
    saved_path = list(sys.path)
    try:
        if sys.path[:1] != [KRAKEN_DIR]:
            sys.path[:] = [KRAKEN_DIR] + [p for p in sys.path if p != KRAKEN_DIR]
        from kraken_client import KrakenClient
        return KrakenClient.public().last_price(pair)
    except Exception:  # noqa: BLE001 — the snapshot must not die on one missed price
        return None
    finally:
        sys.path[:] = saved_path


_f = safe_float


def _is_paper(val) -> bool:
    return _parse_bool_flag(val, False)


def kraken_rows() -> list[dict]:
    rows = []
    for pair, label, paper in KRAKEN_PAIRS:
        s = _load_state(pair)
        if s is None:
            continue
        qty = _f(s.get("qty"))
        cost = _f(s.get("cost"))
        px = _f(_kraken_price(pair), 0.0)
        upnl = (px - cost / qty) * qty if qty > 1e-12 and px > 0 else 0.0
        # Unrealised drawdown as a % of cost (this triggers the risk alert).
        dd_pct = (upnl / cost * 100.0) if cost > 1e-9 else 0.0
        raw_orders = s.get("orders")
        open_orders_cnt = len(raw_orders) if isinstance(raw_orders, (list, tuple, dict)) else 0
        rows.append({
            "engine": f"Kraken {label}" + (" (paper)" if paper else ""),
            "paper": paper,
            "qty": round(qty, 6),
            "value_usd": round(qty * px, 2),
            "cost": round(cost, 2),
            "dd_pct": round(dd_pct, 2),
            "realized_net": round(_f(s.get("realized_net")), 2),
            "unrealized": round(upnl, 2),
            "spent": round(_f(s.get("spent")), 0),
            "cycle": s.get("cycle"),
            "dca_buys": s.get("dca_buys"),
            "open_orders": open_orders_cnt,
        })
    return rows


def hl_row() -> dict | None:
    import contextlib
    import io
    saved_path = list(sys.path)
    saved_common = sys.modules.get("common")
    try:
        hl_dir = os.path.join(ROOT, "hyperliquid")
        if hl_dir not in sys.path:
            sys.path.insert(0, hl_dir)
        sys.modules.pop("common", None)
        # load_dotenv/HLClient write logs to stdout — we suppress them so they do not
        # pollute the --json output (the cron jsonl).
        with contextlib.redirect_stdout(io.StringIO()):
            from common import load_env_stack
            load_env_stack(os.path.join(ROOT, "hyperliquid", ".env"))
            from hl_client import HLClient
            addr = os.environ.get("HL_ACCOUNT_ADDRESS")
            c = HLClient.public(account_address=addr, mainnet=True)
            ss = c.info.spot_user_state(addr)
            mid = _f(c.spot_mid(c.resolve_spot_pair("HYPE")), 0.0)
        raw_balances = ss.get("balances") if isinstance(ss, dict) and isinstance(ss.get("balances"), list) else []
        hype = next((_f(b.get("total")) for b in raw_balances if isinstance(b, dict) and b.get("coin") == "HYPE"), 0.0)
        usdc = next((_f(b.get("total")) for b in raw_balances if isinstance(b, dict) and b.get("coin") == "USDC"), 0.0)
        return {"engine": "HL HYPE (long-hold)", "paper": False, "qty": round(hype, 4),
                "value_usd": round(hype * mid, 2), "cash_usd": round(usdc, 2)}
    except Exception as e:  # noqa: BLE001
        return {"engine": "HL", "error": str(e)[:60]}
    finally:
        sys.path[:] = saved_path
        if saved_common is None:
            sys.modules.pop("common", None)
        else:
            sys.modules["common"] = saved_common


ALERT_STATE = os.path.join(ROOT, "logs", ".risk_alert_state.json")


def _ntfy(title: str, body: str) -> None:
    """Send an ntfy notification (best-effort, no external dependencies)."""
    if any(
        _parse_bool_flag(os.environ.get(k), False)
        for k in ("DISABLE_EXTERNAL_NOTIFICATIONS", "BACKTEST_MODE", "REPLAY_MODE")
    ):
        return
    topic = (os.environ.get("NTFY_TOPIC") or "").strip()
    if not topic:
        # Read from kraken/config.env when it is not in the environment.
        cfg = os.path.join(KRAKEN_DIR, "config.env")
        if os.path.exists(cfg):
            try:
                topic = (parse_dotenv(cfg).get("NTFY_TOPIC") or "").strip()
            except OSError:
                topic = ""
    if not topic:
        return
    import re
    import urllib.request
    clean_title = re.sub(r"[\r\n]+", " ", "" if isinstance(title, bool) else str(title or "Alert")).strip()
    safe_title = clean_title.encode("utf-8").decode("latin-1", "ignore")
    url = topic if topic.startswith(("http://", "https://")) else f"https://ntfy.sh/{topic.lstrip('/')}"
    body_str = "" if isinstance(body, bool) else str(body or "")
    req = urllib.request.Request(
        url, data=body_str.encode("utf-8"),
        headers={"Title": safe_title, "Priority": "high", "Tags": "warning"})
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception:  # noqa: BLE001 — best-effort alert, we do not block the snapshot
        pass


def _alert_state() -> dict:
    try:
        data = read_recovering_previous(ALERT_STATE, json.loads, default=None)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_alert_state(st: dict) -> None:
    try:
        save_json_state(ALERT_STATE, st)
    except (OSError, ValueError, TypeError):
        pass


def run_alerts(kr: list[dict]) -> list[str]:
    """Check the drawdown of every REAL position and send an ntfy alert past the threshold.

    Threshold and re-alert policy are mandatory values from root config.env.
    """
    threshold = _policy_float("RISK_DD_ALERT_PCT")
    realert_hours = _policy_float("RISK_DD_REALERT_HOURS")
    worsen_pct = _policy_float("RISK_DD_WORSEN_PCT")
    now = datetime.now(timezone.utc)
    st = _alert_state()
    fired = []
    for r in kr:
        if not isinstance(r, dict):
            continue
        engine = str(r.get("engine") or "").strip()
        if not engine or _is_paper(r.get("paper")) or _f(r.get("qty"), 0.0) <= 1e-9:
            continue
        dd = _f(r.get("dd_pct"), 0.0)  # Negative means a loss.
        if dd > -threshold:
            st.pop(engine, None)  # Back under the threshold -> reset.
            continue
        prev = st.get(engine, {})
        prev = prev if isinstance(prev, dict) else {}
        prev_dd = _f(prev.get("dd"), 0.0)
        prev_ts = prev.get("ts")
        stale = True
        if prev_ts and not isinstance(prev_ts, bool):
            try:
                parsed_ts = datetime.fromisoformat(str(prev_ts))
                if parsed_ts.tzinfo is None:
                    parsed_ts = parsed_ts.replace(tzinfo=timezone.utc)
                elapsed_s = (now - parsed_ts).total_seconds()
                stale = elapsed_s < 0 or elapsed_s > realert_hours * 3600
            except (TypeError, ValueError):
                stale = True
        if stale or dd <= prev_dd - worsen_pct:
            unreal = _f(r.get("unrealized"), 0.0)
            cost = _f(r.get("cost"), 0.0)
            _ntfy(
                f"⚠️ Risc {engine}: DD {dd:.1f}%",
                f"{engine} drawdown nerealizat {dd:.1f}% "
                f"(nereal ${unreal:.0f} pe cost ${cost:.0f}, "
                f"threshold {threshold:.0f}%). Check the position.")
            st[engine] = {"dd": dd, "ts": now.isoformat(timespec="seconds")}
            fired.append(f"{engine} DD {dd:.1f}%")
    _save_alert_state(st)
    return fired


def main() -> int:
    ap = argparse.ArgumentParser(description="Cross-engine portfolio snapshot (read-only).")
    ap.add_argument("--json", action="store_true", help="one JSON line (for cron/logs)")
    ap.add_argument("--alert", action="store_true",
                    help="check per-position drawdown and send an ntfy alert past the threshold")
    args = ap.parse_args()

    kr = kraken_rows()
    hl = hl_row()
    real_total = sum(_f(r.get("realized_net"), 0.0) for r in kr if isinstance(r, dict) and not _is_paper(r.get("paper")))
    upnl_total = sum(_f(r.get("unrealized"), 0.0) for r in kr if isinstance(r, dict) and not _is_paper(r.get("paper")))
    snap = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "kraken": kr, "hl": hl,
            "kraken_real_realized_net": round(real_total, 2),
            "kraken_real_unrealized": round(upnl_total, 2)}

    if args.alert:
        fired = run_alerts(kr)
        snap["alerts_fired"] = fired

    if args.json:
        print(json.dumps(snap))
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
