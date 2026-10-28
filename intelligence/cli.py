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

from botcore import _parse_bool_flag
from intelligence.macro_sentiment_ssot import (
    get_cache_filenames,
    inspect_daemon_heartbeat,
    load_cached_geopolitical_eval,
    load_cached_sentiment_advisor_eval,
    safe_float as _safe_float,
    safe_int as _safe_int,
)


def _format_age(seconds: Any) -> str:
    """Format duration in human-readable string."""
    sec_f = _safe_float(seconds, float("nan"))
    if not math.isfinite(sec_f):
        return "unknown"
    if sec_f < 0:
        return "now"
    if sec_f < 60:
        return f"{sec_f:.1f}s ago"
    if sec_f < 3600:
        return f"{sec_f / 60:.1f}m ago"
    return f"{sec_f / 3600:.1f}h ago"


def _stamp_age(data: Dict[str, Any], now: float, *, include_seconds: bool = True, ts_fallback_key: Optional[str] = None) -> Dict[str, Any]:
    raw_ts = data.get("ts")
    if raw_ts is None and ts_fallback_key:
        raw_ts = data.get(ts_fallback_key)
    ts = _safe_float(raw_ts, 0.0)
    age = (now - ts) if ts > 0.0 else float("inf")
    if include_seconds:
        data["age_seconds"] = round(age, 1) if math.isfinite(age) else None
    data["age_formatted"] = _format_age(age)
    return data


def _with_formatted_hb_age(res: Dict[str, Any]) -> Dict[str, Any]:
    if "pid" in res:
        age = res.get("age_seconds")
        res["age_formatted"] = _format_age(age if age is not None else float("inf"))
    return res


def get_daemon_status(cache_dir: str = "cachedb") -> Dict[str, Any]:
    """Inspect the running background telemetry daemon."""
    return _with_formatted_hb_age(
        inspect_daemon_heartbeat(
            cache_dir,
            ("globaltelemetry_collector.heartbeat", "intelligence_daemon.heartbeat"),
            max_alive_age_sec=120.0,
            include_symbols=True,
        )
    )


def get_macro_analyzer_status(cache_dir: str = "cachedb") -> Dict[str, Any]:
    """Inspect the running macro intelligence analyzer daemon."""
    return _with_formatted_hb_age(
        inspect_daemon_heartbeat(
            cache_dir,
            ("macro_analyzer.heartbeat",),
            max_alive_age_sec=300.0,
            include_symbols=False,
        )
    )


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

    from state_io import read_recovering_previous

    # 1. Pillar 1: Internal Quantitative & Trend Survival
    try:
        from intelligence.internal.state.survival import _lookup_t_cache_entry, estimate_T
        t_est = None
        if not hasattr(estimate_T, "_mock_name"):
            local_t_data = read_recovering_previous(os.path.join(cache_dir, "cache_T_trend.json"), json.loads, default=None)
            if isinstance(local_t_data, dict):
                t_est = _lookup_t_cache_entry(local_t_data, symbol)
        if not isinstance(t_est, dict):
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

    # 2. Pillar 2: External Microstructure & Derivatives
    try:
        from intelligence.external.collectors.orderbook_depth import _symbol_variants
        sym_candidates = [symbol] if (symbol and "/" not in symbol) else []
        for cand in _symbol_variants(symbol):
            if cand and "/" not in cand and cand not in sym_candidates:
                sym_candidates.append(cand)
        if not sym_candidates:
            sym_candidates = [symbol]

        for key, prefix in (
            ("orderbook", "orderbook_depth"),
            ("derivatives", "derivatives_telemetry"),
            ("whale", "whale_snapshot"),
        ):
            snap_data = None
            for cand in sym_candidates:
                cand_data = read_recovering_previous(os.path.join(cache_dir, f"{prefix}_{cand}.json"), json.loads, default=None)
                if isinstance(cand_data, dict):
                    snap_data = cand_data
                    break
            result["pillar2_external"][key] = _stamp_age(snap_data, now) if isinstance(snap_data, dict) else None
    except Exception as e:
        result["pillar2_external"]["error"] = str(e)

    # 3. Pillar 3: Sentiment & Gemini Advisor
    try:
        fg_data = None
        for fg_fname in get_cache_filenames("fear_greed_files"):
            cand_fg = read_recovering_previous(os.path.join(cache_dir, fg_fname), json.loads, default=None)
            if isinstance(cand_fg, dict):
                fg_data = cand_fg
                break
        result["pillar3_sentiment"]["fear_greed"] = (
            _stamp_age(fg_data, now, include_seconds=False, ts_fallback_key="timestamp")
            if isinstance(fg_data, dict)
            else None
        )

        gem_data = load_cached_sentiment_advisor_eval(cache_dir)
        stamped_gem = _stamp_age(gem_data, now, include_seconds=False) if isinstance(gem_data, dict) else None
        result["pillar3_sentiment"]["sentiment_advisor"] = stamped_gem
        result["pillar3_sentiment"]["gemini_macro"] = stamped_gem
    except Exception as e:
        result["pillar3_sentiment"]["error"] = str(e)

    # 4. Pillar 4: Geopolitical Threat Shield
    try:
        geo_data = load_cached_geopolitical_eval(cache_dir)
        result["pillar4_macro"] = _stamp_age(geo_data, now) if isinstance(geo_data, dict) else None
    except Exception as e:
        result["pillar4_macro_error"] = str(e)

    # 5. Guard Preview (Simulate BUY order evaluation)
    try:
        from order_guard import check_intelligence_guards
        ob_item = result.get("pillar2_external", {}).get("orderbook")
        mid_px_raw = _safe_float(ob_item.get("mid_price"), 100.0) if isinstance(ob_item, dict) else 100.0
        mid_px = mid_px_raw if mid_px_raw > 0.0 else 100.0
        for preview_key, qty_val, notional_val in (("buy_standard", 1.0, 250.0), ("buy_high_stake", 15.0, 1500.0)):
            allowed, reason, scale = check_intelligence_guards(
                provider="binance", symbol=symbol, order_type="BUY", price=mid_px, qty=qty_val, notional_eur=notional_val
            )
            result["guard_preview"][preview_key] = {"allowed": allowed, "reason": reason, "suggested_scale": scale}
    except Exception as e:
        result["guard_preview"]["error"] = str(e)

    return result


def print_intelligence_report(report: Dict[str, Any]) -> None:
    """Print structured, human-readable terminal report."""
    if not isinstance(report, dict):
        report = {}
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
    _as_bool_preview = _parse_bool_flag

    print("\n🚦 [ORDER GUARD EXECUTION PREVIEW (order_guard.py)]")
    buy_std = gp.get("buy_standard")
    if isinstance(buy_std, dict) and buy_std:
        status_std = "✅ ALLOWED" if _as_bool_preview(buy_std.get("allowed")) else "⛔ BLOCKED"
        print(f"  • Standard BUY (250 EUR):   {status_std} | Scale: {_safe_float(buy_std.get('suggested_scale'), 1.0)*100:.0f}% | Reason: {buy_std.get('reason', '')}")
    buy_hi = gp.get("buy_high_stake")
    if isinstance(buy_hi, dict) and buy_hi:
        status_hi = "✅ ALLOWED" if _as_bool_preview(buy_hi.get("allowed")) else "⛔ BLOCKED"
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
