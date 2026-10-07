"""Command-Line Interface (CLI) for Market Intelligence inspection and diagnostics.

Allows operators to inspect real-time multi-pillar state, test order guards,
trigger manual telemetry updates, and check background daemon health.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional


def _format_age(seconds: float) -> str:
    """Format duration in human-readable string."""
    if seconds < 0:
        return "now"
    if seconds < 60:
        return f"{seconds:.1f}s ago"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m ago"
    return f"{seconds / 3600:.1f}h ago"


def get_daemon_status(cache_dir: str = "cachedb") -> Dict[str, Any]:
    """Inspect the running background telemetry daemon."""
    hb_path = os.path.join(cache_dir, "globaltelemetry_collector.heartbeat")
    if not os.path.exists(hb_path):
        hb_path = os.path.join(cache_dir, "intelligence_daemon.heartbeat")
    if not os.path.exists(hb_path):
        return {"running": False, "reason": "no_heartbeat_file"}

    try:
        with open(hb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        pid = int(data.get("pid", 0))
        ts = float(data.get("ts", 0.0))
        age = time.time() - ts

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
            "age_seconds": round(age, 1),
            "age_formatted": _format_age(age),
            "symbols": data.get("symbols", []),
            "intervals": data.get("intervals", {}),
            "last_run": data.get("last_run", {}),
        }
    except Exception as e:
        return {"running": False, "error": str(e)}


def get_macro_analyzer_status(cache_dir: str = "cachedb") -> Dict[str, Any]:
    """Inspect the running macro intelligence analyzer daemon."""
    hb_path = os.path.join(cache_dir, "macro_analyzer.heartbeat")
    if not os.path.exists(hb_path):
        return {"running": False, "reason": "no_heartbeat_file"}

    try:
        with open(hb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        pid = int(data.get("pid", 0))
        ts = float(data.get("ts", 0.0))
        age = time.time() - ts

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
            "age_seconds": round(age, 1),
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

    # 2. Pillar 2: External Microstructure & Derivatives
    try:
        # Orderbook depth
        ob_path = os.path.join(cache_dir, f"orderbook_depth_{symbol}.json")
        if os.path.exists(ob_path):
            with open(ob_path, "r", encoding="utf-8") as f:
                ob_data = json.load(f)
            ob_data["age_seconds"] = round(now - ob_data.get("ts", 0), 1)
            ob_data["age_formatted"] = _format_age(now - ob_data.get("ts", 0))
            result["pillar2_external"]["orderbook"] = ob_data
        else:
            result["pillar2_external"]["orderbook"] = None

        # Derivatives telemetry
        deriv_path = os.path.join(cache_dir, f"derivatives_telemetry_{symbol}.json")
        if os.path.exists(deriv_path):
            with open(deriv_path, "r", encoding="utf-8") as f:
                deriv_data = json.load(f)
            deriv_data["age_seconds"] = round(now - deriv_data.get("ts", 0), 1)
            deriv_data["age_formatted"] = _format_age(now - deriv_data.get("ts", 0))
            result["pillar2_external"]["derivatives"] = deriv_data
        else:
            result["pillar2_external"]["derivatives"] = None

        # Whale positioning
        whale_path = os.path.join(cache_dir, f"whale_snapshot_{symbol}.json")
        if os.path.exists(whale_path):
            with open(whale_path, "r", encoding="utf-8") as f:
                whale_data = json.load(f)
            whale_data["age_seconds"] = round(now - whale_data.get("ts", 0), 1)
            whale_data["age_formatted"] = _format_age(now - whale_data.get("ts", 0))
            result["pillar2_external"]["whale"] = whale_data
        else:
            result["pillar2_external"]["whale"] = None
    except Exception as e:
        result["pillar2_external"]["error"] = str(e)

    # 3. Pillar 3: Sentiment & Gemini Advisor
    try:
        fg_path = os.path.join(cache_dir, "fear_greed_cache.json")
        if os.path.exists(fg_path):
            with open(fg_path, "r", encoding="utf-8") as f:
                fg_data = json.load(f)
            fg_data["age_formatted"] = _format_age(now - fg_data.get("ts", 0))
            result["pillar3_sentiment"]["fear_greed"] = fg_data
        else:
            result["pillar3_sentiment"]["fear_greed"] = None

        gemini_path = os.path.join(cache_dir, "gemini_macro_advisor.json")
        if os.path.exists(gemini_path):
            with open(gemini_path, "r", encoding="utf-8") as f:
                gem_data = json.load(f)
            gem_data["age_formatted"] = _format_age(now - gem_data.get("ts", 0))
            result["pillar3_sentiment"]["gemini_macro"] = gem_data
        else:
            result["pillar3_sentiment"]["gemini_macro"] = None
    except Exception as e:
        result["pillar3_sentiment"]["error"] = str(e)

    # 4. Pillar 4: Geopolitical Threat Shield
    try:
        geo_path = os.path.join(cache_dir, "geopolitical_threat_state.json")
        if os.path.exists(geo_path):
            with open(geo_path, "r", encoding="utf-8") as f:
                geo_data = json.load(f)
            geo_data["age_seconds"] = round(now - geo_data.get("ts", 0), 1)
            geo_data["age_formatted"] = _format_age(now - geo_data.get("ts", 0))
            result["pillar4_macro"] = geo_data
        else:
            result["pillar4_macro"] = None
    except Exception as e:
        result["pillar4_macro_error"] = str(e)

    # 5. Guard Preview (Simulate BUY order evaluation)
    try:
        from order_guard import check_intelligence_guards
        ob_item = result.get("pillar2_external", {}).get("orderbook")
        mid_px = float(ob_item.get("mid_price", 100.0)) if ob_item else 100.0
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
