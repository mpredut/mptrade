"""Command-Line Interface (CLI) for Market Intelligence inspection and diagnostics.

Allows operators to inspect real-time multi-pillar state, test order guards,
trigger manual telemetry updates, and check background daemon health.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, Optional


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None or isinstance(val, bool):
        return default
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError, OverflowError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    if val is None or isinstance(val, bool):
        return default
    try:
        f = float(val)
        return int(f) if math.isfinite(f) else default
    except (TypeError, ValueError, OverflowError):
        return default


def _format_age(seconds: float) -> str:
    """Format duration in human-readable string."""
    if seconds is None or isinstance(seconds, bool):
        return "unknown"
    try:
        sec_f = float(seconds)
    except (TypeError, ValueError, OverflowError):
        return "unknown"
    if not math.isfinite(sec_f):
        return "unknown"
    if sec_f < 0:
        return "now"
    if sec_f < 60:
        return f"{sec_f:.1f}s ago"
    if sec_f < 3600:
        return f"{sec_f / 60:.1f}m ago"
    return f"{sec_f / 3600:.1f}h ago"


def get_daemon_status(cache_dir: str = "cachedb") -> Dict[str, Any]:
    """Inspect the running background telemetry daemon."""
    from state_io import read_recovering_previous
    data = None
    for fname in ("globaltelemetry_collector.heartbeat", "intelligence_daemon.heartbeat"):
        cand = os.path.join(cache_dir, fname)
        if os.path.exists(cand) or os.path.exists(f"{cand}.previous"):
            try:
                cand_data = read_recovering_previous(cand, json.loads, default=None)
                if isinstance(cand_data, dict):
                    data = cand_data
                    break
            except Exception:
                continue
    if data is None:
        return {"running": False, "reason": "no_heartbeat_file"}

    try:
        pid = _safe_int(data.get("pid", 0), 0)
        ts = _safe_float(data.get("ts", 0.0), 0.0)
        age = max(0.0, time.time() - ts) if ts > 0.0 else float("inf")

        is_alive = False
        if pid > 0:
            try:
                # Check process existence via signal 0
                os.kill(pid, 0)
                is_alive = True
            except PermissionError:
                is_alive = True
            except (OSError, OverflowError):
                is_alive = False

        status_str = "alive" if (is_alive and age < 120.0) else ("stale" if is_alive else "dead")
        raw_symbols = data.get("symbols")
        symbols_list = [str(s) for s in raw_symbols if s is not None and not isinstance(s, bool)] if isinstance(raw_symbols, list) else []
        raw_intervals = data.get("intervals")
        raw_last_run = data.get("last_run")
        return {
            "running": is_alive and age < 120.0,
            "pid": pid,
            "status": status_str,
            "age_seconds": round(age, 1) if math.isfinite(age) else None,
            "age_formatted": _format_age(age),
            "symbols": symbols_list,
            "intervals": raw_intervals if isinstance(raw_intervals, dict) else {},
            "last_run": raw_last_run if isinstance(raw_last_run, dict) else {},
        }
    except Exception as e:
        return {"running": False, "error": str(e)}


def get_macro_analyzer_status(cache_dir: str = "cachedb") -> Dict[str, Any]:
    """Inspect the running macro intelligence analyzer daemon."""
    from state_io import read_recovering_previous
    hb_path = os.path.join(cache_dir, "macro_analyzer.heartbeat")
    if not (os.path.exists(hb_path) or os.path.exists(f"{hb_path}.previous")):
        return {"running": False, "reason": "no_heartbeat_file"}

    try:
        data = read_recovering_previous(hb_path, json.loads, default=None)
        if not isinstance(data, dict):
            return {"running": False, "reason": "no_heartbeat_file"}
        pid = _safe_int(data.get("pid", 0), 0)
        ts = _safe_float(data.get("ts", 0.0), 0.0)
        age = max(0.0, time.time() - ts) if ts > 0.0 else float("inf")

        is_alive = False
        if pid > 0:
            try:
                os.kill(pid, 0)
                is_alive = True
            except PermissionError:
                is_alive = True
            except (OSError, OverflowError):
                is_alive = False

        status_str = "alive" if (is_alive and age < 300.0) else ("stale" if is_alive else "dead")
        raw_intervals = data.get("intervals")
        raw_last_run = data.get("last_run")
        return {
            "running": is_alive and age < 300.0,
            "pid": pid,
            "status": status_str,
            "age_seconds": round(age, 1) if math.isfinite(age) else None,
            "age_formatted": _format_age(age),
            "intervals": raw_intervals if isinstance(raw_intervals, dict) else {},
            "last_run": raw_last_run if isinstance(raw_last_run, dict) else {},
        }
    except Exception as e:
        return {"running": False, "error": str(e)}


def inspect_symbol_intelligence(symbol: str, cache_dir: str = "cachedb") -> Dict[str, Any]:
    """Consolidate full multi-pillar state for a symbol."""
    symbol = "" if isinstance(symbol, bool) or symbol is None else str(symbol).strip().upper()
    now = time.time()
    result: Dict[str, Any] = {
        "symbol": symbol,
        "ts": now,
        "pillar1_internal": {},
        "pillar2_external": {},
        "pillar3_sentiment": {},
        "pillar4_macro": {},
        "guard_preview": {},
    }

    # 1. Pillar 1: Internal Quantitative & Trend Survival
    try:
        from intelligence.internal.state.survival import estimate_T
        t_est = estimate_T(symbol)
        samples_n = max(0, _safe_int(t_est.get("n", 0), 0)) if isinstance(t_est, dict) else 0
        result["pillar1_internal"]["trend_survival"] = {
            "T": t_est.get("T", 14) if isinstance(t_est, dict) else 14,
            "T_emp": t_est.get("T_emp") if isinstance(t_est, dict) else None,
            "median_duration_days": t_est.get("median_d") if isinstance(t_est, dict) else None,
            "P90_days": t_est.get("p90_d") if isinstance(t_est, dict) else None,
            "samples": samples_n,
            "is_calibrated": samples_n > 0,
        }
    except Exception as e:
        result["pillar1_internal"]["trend_survival_error"] = str(e)

    from state_io import read_recovering_previous

    # 2. Pillar 2: External Microstructure & Derivatives
    try:
        # Orderbook depth
        ob_path = os.path.join(cache_dir, f"orderbook_depth_{symbol}.json")
        ob_data = read_recovering_previous(ob_path, json.loads, default=None)
        if isinstance(ob_data, dict):
            ob_ts = _safe_float(ob_data.get("ts"), 0.0)
            ob_age = (now - ob_ts) if ob_ts > 0.0 else float("inf")
            ob_data["age_seconds"] = round(ob_age, 1) if math.isfinite(ob_age) else None
            ob_data["age_formatted"] = _format_age(ob_age)
            result["pillar2_external"]["orderbook"] = ob_data
        else:
            result["pillar2_external"]["orderbook"] = None

        # Derivatives telemetry
        deriv_path = os.path.join(cache_dir, f"derivatives_telemetry_{symbol}.json")
        deriv_data = read_recovering_previous(deriv_path, json.loads, default=None)
        if isinstance(deriv_data, dict):
            deriv_ts = _safe_float(deriv_data.get("ts"), 0.0)
            deriv_age = (now - deriv_ts) if deriv_ts > 0.0 else float("inf")
            deriv_data["age_seconds"] = round(deriv_age, 1) if math.isfinite(deriv_age) else None
            deriv_data["age_formatted"] = _format_age(deriv_age)
            result["pillar2_external"]["derivatives"] = deriv_data
        else:
            result["pillar2_external"]["derivatives"] = None

        # Whale positioning
        whale_path = os.path.join(cache_dir, f"whale_snapshot_{symbol}.json")
        whale_data = read_recovering_previous(whale_path, json.loads, default=None)
        if isinstance(whale_data, dict):
            whale_ts = _safe_float(whale_data.get("ts"), 0.0)
            whale_age = (now - whale_ts) if whale_ts > 0.0 else float("inf")
            whale_data["age_seconds"] = round(whale_age, 1) if math.isfinite(whale_age) else None
            whale_data["age_formatted"] = _format_age(whale_age)
            result["pillar2_external"]["whale"] = whale_data
        else:
            result["pillar2_external"]["whale"] = None
    except Exception as e:
        result["pillar2_external"]["error"] = str(e)

    # 3. Pillar 3: Sentiment & Gemini Advisor
    try:
        fg_data = None
        for fg_fname in ("fear_greed_collect.json", "fear_greed_cache.json"):
            cand_fg = read_recovering_previous(os.path.join(cache_dir, fg_fname), json.loads, default=None)
            if isinstance(cand_fg, dict):
                fg_data = cand_fg
                break
        if isinstance(fg_data, dict):
            raw_fg_ts = fg_data.get("ts") if fg_data.get("ts") is not None else fg_data.get("timestamp")
            fg_ts = _safe_float(raw_fg_ts, 0.0)
            fg_age = (now - fg_ts) if fg_ts > 0.0 else float("inf")
            fg_data["age_formatted"] = _format_age(fg_age)
            result["pillar3_sentiment"]["fear_greed"] = fg_data
        else:
            result["pillar3_sentiment"]["fear_greed"] = None

        gem_data = None
        for adv_fname in ("sentiment_advisor_eval.json", "macro_advisor_eval.json", "gemini_macro_advisor.json"):
            cand_adv = read_recovering_previous(os.path.join(cache_dir, adv_fname), json.loads, default=None)
            if isinstance(cand_adv, dict):
                gem_data = cand_adv
                break
        if isinstance(gem_data, dict):
            gem_ts = _safe_float(gem_data.get("ts"), 0.0)
            gem_age = (now - gem_ts) if gem_ts > 0.0 else float("inf")
            gem_data["age_formatted"] = _format_age(gem_age)
            result["pillar3_sentiment"]["sentiment_advisor"] = gem_data
            result["pillar3_sentiment"]["gemini_macro"] = gem_data
        else:
            result["pillar3_sentiment"]["sentiment_advisor"] = None
            result["pillar3_sentiment"]["gemini_macro"] = None
    except Exception as e:
        result["pillar3_sentiment"]["error"] = str(e)

    # 4. Pillar 4: Geopolitical Threat Shield
    try:
        geo_data = None
        for geo_fname in ("geopolitical_threat_eval.json", "geopolitical_threat_state.json"):
            cand_geo = read_recovering_previous(os.path.join(cache_dir, geo_fname), json.loads, default=None)
            if isinstance(cand_geo, dict):
                geo_data = cand_geo
                break
        if isinstance(geo_data, dict):
            geo_ts = _safe_float(geo_data.get("ts"), 0.0)
            geo_age = (now - geo_ts) if geo_ts > 0.0 else float("inf")
            geo_data["age_seconds"] = round(geo_age, 1) if math.isfinite(geo_age) else None
            geo_data["age_formatted"] = _format_age(geo_age)
            result["pillar4_macro"] = geo_data
        else:
            result["pillar4_macro"] = None
    except Exception as e:
        result["pillar4_macro_error"] = str(e)

    # 5. Guard Preview (Simulate BUY order evaluation)
    try:
        from order_guard import check_intelligence_guards
        ob_item = result.get("pillar2_external", {}).get("orderbook")
        mid_px_raw = _safe_float(ob_item.get("mid_price"), 100.0) if isinstance(ob_item, dict) else 100.0
        mid_px = mid_px_raw if mid_px_raw > 0.0 else 100.0
        # BUY evaluation with standard 250 EUR notional
        allowed, reason, scale = check_intelligence_guards(
            provider="binance",
            symbol=symbol,
            order_type="BUY",
            price=mid_px,
            qty=1.0,
            notional_eur=250.0,
        )
        result["guard_preview"]["buy_standard"] = {
            "allowed": allowed,
            "reason": reason,
            "suggested_scale": scale,
        }

        # BUY evaluation with high stake (1500 EUR)
        allowed_high, reason_high, scale_high = check_intelligence_guards(
            provider="binance",
            symbol=symbol,
            order_type="BUY",
            price=mid_px,
            qty=15.0,
            notional_eur=1500.0,
        )
        result["guard_preview"]["buy_high_stake"] = {
            "allowed": allowed_high,
            "reason": reason_high,
            "suggested_scale": scale_high,
        }
    except Exception as e:
        result["guard_preview"]["error"] = str(e)

    return result


def print_intelligence_report(report: Dict[str, Any]) -> None:
    """Print structured, human-readable terminal report."""
    symbol = str(report.get("symbol") or "UNKNOWN")
    raw_p1 = report.get("pillar1_internal")
    p1 = raw_p1 if isinstance(raw_p1, dict) else {}
    raw_p2 = report.get("pillar2_external")
    p2 = raw_p2 if isinstance(raw_p2, dict) else {}
    raw_p3 = report.get("pillar3_sentiment")
    p3 = raw_p3 if isinstance(raw_p3, dict) else {}
    raw_p4 = report.get("pillar4_macro")
    p4 = raw_p4 if isinstance(raw_p4, dict) else {}
    raw_gp = report.get("guard_preview")
    gp = raw_gp if isinstance(raw_gp, dict) else {}

    print("\n" + "=" * 76)
    print(f"  🔍 MARKET INTELLIGENCE AUDIT — {symbol}")
    print("=" * 76)

    # Pillar 1
    print("\n📊 [PILLAR 1: INTERNAL QUANTITATIVE & SURVIVAL]")
    raw_surv = p1.get("trend_survival")
    surv = raw_surv if isinstance(raw_surv, dict) else None
    if surv and surv.get("P90_days") is not None:
        print(f"  • Trend Survival: T_emp={surv.get('T_emp')}d | Median={surv.get('median_duration_days')}d | P90={surv.get('P90_days')}d")
        print(f"  • Calibration:    {surv.get('samples', 0)} empirical trend cycles")
    elif surv:
        print(f"  • Trend Survival: Prior default T={surv.get('T', 14)}d (samples: {surv.get('samples', 0)})")
    else:
        print("  • Trend Survival: None")

    # Pillar 2
    print("\n🌊 [PILLAR 2: EXTERNAL MICROSTRUCTURE & DERIVATIVES FLOW]")
    ob = p2.get("orderbook")
    if isinstance(ob, dict) and ob:
        imbalance = _safe_float(ob.get("imbalance_ratio"), 0.5)
        imb_bar = "🟢 BIDS" if imbalance > 0.55 else ("🔴 ASKS" if imbalance < 0.45 else "⚪ BALANCED")
        print(f"  • Spot Depth:     Mid=${_safe_float(ob.get('mid_price'), 0.0):,.2f} | Bids=${_safe_float(ob.get('bid_depth_usd'), 0.0):,.0f} | Asks=${_safe_float(ob.get('ask_depth_usd'), 0.0):,.0f}")
        print(f"  • Imbalance:      {imbalance:.2f} ({imb_bar}) [snapshot: {ob.get('age_formatted', 'unknown')}]")
        print(f"  • Max Bid Wall:   ${_safe_float(ob.get('largest_bid_wall_usd'), 0.0):,.0f} at ${_safe_float(ob.get('largest_bid_wall_price'), 0.0):,.2f}")
        print(f"  • Max Ask Wall:   ${_safe_float(ob.get('largest_ask_wall_usd'), 0.0):,.0f} at ${_safe_float(ob.get('largest_ask_wall_price'), 0.0):,.2f}")
    else:
        print("  • Spot Depth:     [No snapshot in cachedb]")

    deriv = p2.get("derivatives")
    if isinstance(deriv, dict) and deriv:
        funding = _safe_float(deriv.get("funding_rate"), 0.0)
        crowding = "⚠️ CROWDED LONG" if funding > 0.0005 else ("🔥 NEGATIVE/SQUEEZE" if funding < -0.0002 else "NORMAL")
        print(f"  • Funding Rate:   {funding:+.6f} ({funding*100:+.4f}% / 8h) [{crowding}]")
        print(f"  • Open Interest:  ${_safe_float(deriv.get('open_interest_usd'), 0.0):,.0f} contracts [snapshot: {deriv.get('age_formatted', 'unknown')}]")
    else:
        print("  • Derivatives:    [No snapshot in cachedb]")

    whale = p2.get("whale")
    if isinstance(whale, dict) and whale:
        regime = str(whale.get("divergence_regime") or "neutral").upper()
        print(f"  • Whale L/S:      {_safe_float(whale.get('top_traders_long_ratio'), 1.0):.2f} ({_safe_float(whale.get('top_traders_long_pct'), 0.5)*100:.1f}% longs)")
        print(f"  • Taker Vol Buy:  ${_safe_float(whale.get('taker_buy_vol_usd'), 0.0):,.0f} vs Sell: ${_safe_float(whale.get('taker_sell_vol_usd'), 0.0):,.0f} (Ratio: {_safe_float(whale.get('taker_buy_sell_ratio'), 1.0):.2f})")
        print(f"  • OI Divergence:  Regime={regime} | OI 1h Delta={_safe_float(whale.get('open_interest_1h_change_pct'), 0.0):+.2f}% [snapshot: {whale.get('age_formatted', 'unknown')}]")
    else:
        print("  • Whale Flow:     [No snapshot in cachedb]")

    # Pillar 3
    print("\n🧠 [PILLAR 3: SENTIMENT & GEMINI LLM REASONING]")
    fg = p3.get("fear_greed")
    if isinstance(fg, dict) and fg:
        fg_label = fg.get("sentiment") or fg.get("classification") or "Neutral"
        print(f"  • Fear & Greed:   {_safe_int(fg.get('value'), 50)}/100 ({fg_label}) [snapshot: {fg.get('age_formatted', 'unknown')}]")
    else:
        print("  • Fear & Greed:   [No snapshot in cachedb]")

    gem = p3.get("gemini_macro")
    if isinstance(gem, dict) and gem:
        print(f"  • Gemini Advisor: {gem.get('summary', 'No summary')} [snapshot: {gem.get('age_formatted', 'unknown')}]")
    else:
        print("  • Gemini Advisor: [No snapshot in cachedb]")

    # Pillar 4
    print("\n🛡️ [PILLAR 4: MACRO GEOPOLITICAL & ENERGY SHOCK SHIELD]")
    if isinstance(p4, dict) and p4:
        level = str(p4.get("threat_level") or "NORMAL")
        icon = "🟢" if level == "NORMAL" else ("🟡" if level == "ELEVATED" else "🔴")
        print(f"  • Threat Level:   {icon} {level} (Risk Score: {_safe_float(p4.get('risk_score'), 0.0):.2f}) [snapshot: {p4.get('age_formatted', 'unknown')}]")
        print(f"  • Brake Action:   {p4.get('recommended_brake', 'NONE')}")
        print(f"  • Analysis:       {p4.get('summary', '')}")
    else:
        print("  • Macro Threat:   [No threat assessment in cachedb]")

    # Order Guard Preview
    print("\n🚦 [ORDER GUARD EXECUTION PREVIEW (order_guard.py)]")
    buy_std = gp.get("buy_standard")
    if isinstance(buy_std, dict) and buy_std:
        status_std = "✅ ALLOWED" if buy_std.get("allowed") else "⛔ BLOCKED"
        print(f"  • Standard BUY (250 EUR):   {status_std} | Scale: {_safe_float(buy_std.get('suggested_scale'), 1.0)*100:.0f}% | Reason: {buy_std.get('reason', '')}")
    buy_hi = gp.get("buy_high_stake")
    if isinstance(buy_hi, dict) and buy_hi:
        status_hi = "✅ ALLOWED" if buy_hi.get("allowed") else "⛔ BLOCKED"
        print(f"  • High-Stake BUY (1500 EUR): {status_hi} | Scale: {_safe_float(buy_hi.get('suggested_scale'), 1.0)*100:.0f}% | Reason: {buy_hi.get('reason', '')}")

    print("=" * 76 + "\n")


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Market Intelligence Interactive CLI & Diagnostics")
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDC",
        help="Target symbol to inspect (default: BTCUSDC).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Trigger immediate collection and refresh of snapshots before inspecting.",
    )
    parser.add_argument(
        "--daemon-status",
        action="store_true",
        help="Inspect the background telemetry collector daemon health.",
    )
    parser.add_argument(
        "--test-guard",
        action="store_true",
        help="Run an interactive order guard simulation.",
    )
    parser.add_argument(
        "--side",
        type=str,
        default="BUY",
        choices=["BUY", "SELL"],
        help="Order side for guard test (default: BUY).",
    )
    parser.add_argument(
        "--qty",
        type=float,
        default=1.0,
        help="Order quantity for guard test (default: 1.0).",
    )
    parser.add_argument(
        "--notional",
        type=float,
        default=250.0,
        help="Order notional in EUR for guard test (default: 250.0).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw machine-readable JSON format.",
    )

    args = parser.parse_args()
    symbol = args.symbol.strip().upper()

    # 1. Daemon Status Check
    if args.daemon_status:
        status = get_daemon_status()
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print("\n" + "=" * 60)
            print("  📡 MARKET INTELLIGENCE TELEMETRY DAEMON STATUS")
            print("=" * 60)
            print(f"  • State:       {status.get('status', 'unknown').upper()}")
            print(f"  • PID:         {status.get('pid', 'none')}")
            print(f"  • Heartbeat:   {status.get('age_formatted', 'never')} ({status.get('age_seconds', 0)}s)")
            print(f"  • Symbols:     {', '.join(status.get('symbols', []))}")
            print("=" * 60 + "\n")
        return

    # 2. Optional Manual Refresh
    if args.refresh:
        from intelligence.daemon import IntelligenceTelemetryDaemon
        print(f"Refreshing telemetry for {symbol}...")
        daemon = IntelligenceTelemetryDaemon(symbols=[symbol])
        res = daemon.run_cycle(force=True)
        if not args.json:
            print(f"Refresh completed: {res}")

    # 3. Test Guard Simulation
    if args.test_guard:
        from order_guard import check_intelligence_guards
        allowed, reason, scale = check_intelligence_guards(
            provider="binance",
            symbol=symbol,
            order_type=args.side,
            price=100.0,
            qty=args.qty,
            notional_eur=args.notional,
        )
        guard_res = {
            "symbol": symbol,
            "side": args.side,
            "qty": args.qty,
            "notional_eur": args.notional,
            "allowed": allowed,
            "reason": reason,
            "suggested_scale": scale,
        }
        if args.json:
            print(json.dumps(guard_res, indent=2))
        else:
            print("\n" + "=" * 60)
            print(f"  🛡️ TEST GUARD SIMULATION — {args.side} {symbol}")
            print("=" * 60)
            print(f"  • Allowed:          {'YES' if allowed else 'NO'}")
            print(f"  • Suggested Scale:  {scale * 100:.0f}%")
            print(f"  • Reason:           {reason}")
            print("=" * 60 + "\n")
        return

    # 4. Standard Inspection Report
    report = inspect_symbol_intelligence(symbol)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_intelligence_report(report)


if __name__ == "__main__":
    main()
