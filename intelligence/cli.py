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
        age = (time.time() - ts) if ts > 0.0 else float("inf")

        is_alive = False
        if pid > 0:
            try:
                # Check process existence via signal 0
                os.kill(pid, 0)
                is_alive = True
            except OSError:
                is_alive = False

        status_str = "alive" if (is_alive and age < 120.0) else ("stale" if is_alive else "dead")
        return {
            "running": is_alive and age < 120.0,
            "pid": pid,
            "status": status_str,
            "age_seconds": round(age, 1) if math.isfinite(age) else None,
            "age_formatted": _format_age(age),
            "symbols": data.get("symbols", []),
            "intervals": data.get("intervals", {}),
            "last_run": data.get("last_run", {}),
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
        age = (time.time() - ts) if ts > 0.0 else float("inf")

        is_alive = False
        if pid > 0:
            try:
                os.kill(pid, 0)
                is_alive = True
            except OSError:
                is_alive = False

        status_str = "alive" if (is_alive and age < 300.0) else ("stale" if is_alive else "dead")
        return {
            "running": is_alive and age < 300.0,
            "pid": pid,
            "status": status_str,
            "age_seconds": round(age, 1) if math.isfinite(age) else None,
            "age_formatted": _format_age(age),
            "intervals": data.get("intervals", {}),
            "last_run": data.get("last_run", {}),
        }
    except Exception as e:
        return {"running": False, "error": str(e)}


def inspect_symbol_intelligence(symbol: str, cache_dir: str = "cachedb") -> Dict[str, Any]:
    """Consolidate full multi-pillar state for a symbol."""
    symbol = symbol.upper()
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
        result["pillar1_internal"]["trend_survival"] = {
            "T": t_est.get("T", 14),
            "T_emp": t_est.get("T_emp"),
            "median_duration_days": t_est.get("median_d"),
            "P90_days": t_est.get("p90_d"),
            "samples": t_est.get("n", 0),
            "is_calibrated": t_est.get("n", 0) > 0,
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
            ob_data["age_seconds"] = round(now - ob_ts, 1)
            ob_data["age_formatted"] = _format_age(now - ob_ts)
            result["pillar2_external"]["orderbook"] = ob_data
        else:
            result["pillar2_external"]["orderbook"] = None

        # Derivatives telemetry
        deriv_path = os.path.join(cache_dir, f"derivatives_telemetry_{symbol}.json")
        deriv_data = read_recovering_previous(deriv_path, json.loads, default=None)
        if isinstance(deriv_data, dict):
            deriv_ts = _safe_float(deriv_data.get("ts"), 0.0)
            deriv_data["age_seconds"] = round(now - deriv_ts, 1)
            deriv_data["age_formatted"] = _format_age(now - deriv_ts)
            result["pillar2_external"]["derivatives"] = deriv_data
        else:
            result["pillar2_external"]["derivatives"] = None

        # Whale positioning
        whale_path = os.path.join(cache_dir, f"whale_snapshot_{symbol}.json")
        whale_data = read_recovering_previous(whale_path, json.loads, default=None)
        if isinstance(whale_data, dict):
            whale_ts = _safe_float(whale_data.get("ts"), 0.0)
            whale_data["age_seconds"] = round(now - whale_ts, 1)
            whale_data["age_formatted"] = _format_age(now - whale_ts)
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
            fg_ts = _safe_float(fg_data.get("ts"), 0.0)
            fg_data["age_formatted"] = _format_age(now - fg_ts)
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
            gem_data["age_formatted"] = _format_age(now - gem_ts)
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
            geo_data["age_seconds"] = round(now - geo_ts, 1)
            geo_data["age_formatted"] = _format_age(now - geo_ts)
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
    symbol = report["symbol"]
    p1 = report.get("pillar1_internal", {})
    p2 = report.get("pillar2_external", {})
    p3 = report.get("pillar3_sentiment", {})
    p4 = report.get("pillar4_macro", {})
    gp = report.get("guard_preview", {})

    print("\n" + "=" * 76)
    print(f"  🔍 MARKET INTELLIGENCE AUDIT — {symbol}")
    print("=" * 76)

    # Pillar 1
    print("\n📊 [PILLAR 1: INTERNAL QUANTITATIVE & SURVIVAL]")
    surv = p1.get("trend_survival")
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
    if ob:
        imbalance = ob.get("imbalance_ratio", 0.5)
        imb_bar = "🟢 BIDS" if imbalance > 0.55 else ("🔴 ASKS" if imbalance < 0.45 else "⚪ BALANCED")
        print(f"  • Spot Depth:     Mid=${ob.get('mid_price', 0):,.2f} | Bids=${ob.get('bid_depth_usd', 0):,.0f} | Asks=${ob.get('ask_depth_usd', 0):,.0f}")
        print(f"  • Imbalance:      {imbalance:.2f} ({imb_bar}) [snapshot: {ob.get('age_formatted', 'unknown')}]")
        print(f"  • Max Bid Wall:   ${ob.get('largest_bid_wall_usd', 0):,.0f} at ${ob.get('largest_bid_wall_price', 0):,.2f}")
        print(f"  • Max Ask Wall:   ${ob.get('largest_ask_wall_usd', 0):,.0f} at ${ob.get('largest_ask_wall_price', 0):,.2f}")
    else:
        print("  • Spot Depth:     [No snapshot in cachedb]")

    deriv = p2.get("derivatives")
    if deriv:
        funding = deriv.get("funding_rate", 0.0)
        crowding = "⚠️ CROWDED LONG" if funding > 0.0005 else ("🔥 NEGATIVE/SQUEEZE" if funding < -0.0002 else "NORMAL")
        print(f"  • Funding Rate:   {funding:+.6f} ({funding*100:+.4f}% / 8h) [{crowding}]")
        print(f"  • Open Interest:  ${deriv.get('open_interest_usd', 0):,.0f} contracts [snapshot: {deriv.get('age_formatted', 'unknown')}]")
    else:
        print("  • Derivatives:    [No snapshot in cachedb]")

    whale = p2.get("whale")
    if whale:
        regime = whale.get("divergence_regime", "neutral").upper()
        print(f"  • Whale L/S:      {whale.get('top_traders_long_ratio', 1.0):.2f} ({whale.get('top_traders_long_pct', 0.5)*100:.1f}% longs)")
        print(f"  • Taker Vol Buy:  ${whale.get('taker_buy_vol_usd', 0):,.0f} vs Sell: ${whale.get('taker_sell_vol_usd', 0):,.0f} (Ratio: {whale.get('taker_buy_sell_ratio', 1.0):.2f})")
        print(f"  • OI Divergence:  Regime={regime} | OI 1h Delta={whale.get('open_interest_1h_change_pct', 0.0):+.2f}% [snapshot: {whale.get('age_formatted', 'unknown')}]")
    else:
        print("  • Whale Flow:     [No snapshot in cachedb]")

    # Pillar 3
    print("\n🧠 [PILLAR 3: SENTIMENT & GEMINI LLM REASONING]")
    fg = p3.get("fear_greed")
    if fg:
        print(f"  • Fear & Greed:   {fg.get('value', 50)}/100 ({fg.get('sentiment', 'Neutral')}) [snapshot: {fg.get('age_formatted', 'unknown')}]")
    else:
        print("  • Fear & Greed:   [No snapshot in cachedb]")

    gem = p3.get("gemini_macro")
    if gem:
        print(f"  • Gemini Advisor: {gem.get('summary', 'No summary')} [snapshot: {gem.get('age_formatted', 'unknown')}]")
    else:
        print("  • Gemini Advisor: [No snapshot in cachedb]")

    # Pillar 4
    print("\n🛡️ [PILLAR 4: MACRO GEOPOLITICAL & ENERGY SHOCK SHIELD]")
    if p4:
        level = p4.get("threat_level", "NORMAL")
        icon = "🟢" if level == "NORMAL" else ("🟡" if level == "ELEVATED" else "🔴")
        print(f"  • Threat Level:   {icon} {level} (Risk Score: {p4.get('risk_score', 0.0):.2f}) [snapshot: {p4.get('age_formatted', 'unknown')}]")
        print(f"  • Brake Action:   {p4.get('recommended_brake', 'NONE')}")
        print(f"  • Analysis:       {p4.get('summary', '')}")
    else:
        print("  • Macro Threat:   [No threat assessment in cachedb]")

    # Order Guard Preview
    print("\n🚦 [ORDER GUARD EXECUTION PREVIEW (order_guard.py)]")
    buy_std = gp.get("buy_standard")
    if buy_std:
        status_std = "✅ ALLOWED" if buy_std["allowed"] else "⛔ BLOCKED"
        print(f"  • Standard BUY (250 EUR):   {status_std} | Scale: {buy_std['suggested_scale']*100:.0f}% | Reason: {buy_std['reason']}")
    buy_hi = gp.get("buy_high_stake")
    if buy_hi:
        status_hi = "✅ ALLOWED" if buy_hi["allowed"] else "⛔ BLOCKED"
        print(f"  • High-Stake BUY (1500 EUR): {status_hi} | Scale: {buy_hi['suggested_scale']*100:.0f}% | Reason: {buy_hi['reason']}")

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
