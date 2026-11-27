#!/usr/bin/env python3
"""Audit script to verify the quality of telemetry and data sent in auto_cryptotrade_executor prompt."""

import os
import re
import sys
import time
from typing import Any, Dict, List

from botcore import canonical_topic_key, topic_spec
from state_io import encode_json_payload
from orchestratorOS.lib.technical_repair_ssot import LogEventSummary
from orchestratorOS.supervisor.autonomous_ai_reconciler import (
    AutonomousAIReconciler,
)
from orchestratorOS.supervisor.prompt_formatters import build_supervisor_prompt


def inspect_crypto_prompt(workspace_dir: str = ".") -> Dict[str, Any]:
    topic_key = "auto_cryptotrade_executor"
    spec = topic_spec(topic_key)
    reconciler = AutonomousAIReconciler(
        mode="shadow",
        workspace_dir=workspace_dir,
        cache_dir=os.path.join(workspace_dir, "cachedb"),
        logs_dirs=[os.path.join(workspace_dir, "logs")],
    )

    sensors = reconciler.harvest_sensors_telemetry(
        include_fleet=spec.can_self_heal,
        include_market_portfolio=spec.can_trade,
        asset_scope=spec.asset_scope,
        allow_network=False,
        persist=False,
    )
    logs = LogEventSummary(total_scanned_files=0, error_count=0, warning_count=0)

    prompt = build_supervisor_prompt(
        sensors,
        logs,
        spec,
        reconciler.lookback_hours,
        focus_symbols=None,
        min_usd=10.0,
        max_usd=2000.0,
        min_conf=0.85,
        compact_rules=False,
    )

    resume_prompt = build_supervisor_prompt(
        sensors,
        logs,
        spec,
        reconciler.lookback_hours,
        focus_symbols=None,
        min_usd=10.0,
        max_usd=2000.0,
        min_conf=0.85,
        compact_rules=True,
    )

    prompt_len = len(prompt)
    est_tokens = prompt_len // 4

    # Analyze prompt sections
    sections = {}
    current_sec = "HEADER"
    lines = prompt.split("\n")
    sec_lines: Dict[str, List[str]] = {current_sec: []}

    for line in lines:
        if line.startswith("### "):
            current_sec = line.strip()
            sec_lines[current_sec] = []
        elif line.startswith("INTERACTIVE THREAD RULES:") or line.startswith("CONTINUATION TURN"):
            current_sec = line.strip()
            sec_lines[current_sec] = []
        elif line.startswith("LIVE TELEMETRY SNAPSHOT:"):
            current_sec = "LIVE TELEMETRY SNAPSHOT"
            sec_lines[current_sec] = []
        else:
            sec_lines[current_sec].append(line)

    # Detect data quality anomalies
    na_count = len(re.findall(r"\bN/A\b", prompt))
    none_count = len(re.findall(r"\bNone\b", prompt))
    nan_count = len(re.findall(r"\bnan\b|\bNaN\b", prompt))
    stale_count = len(re.findall(r"\[stale\]|STALE_FROZEN|\[ob_stale\]", prompt))

    # Check for specific expected indicators derived dynamically from instruments.conf SSOT
    from instrument_registry import is_equity_symbol, tracked_price_instruments
    all_tracked, _ = tracked_price_instruments()
    expected_symbols = [sym for sym in all_tracked if not is_equity_symbol(sym)]
    found_symbols = {sym: (sym in prompt) for sym in expected_symbols}

    key_features = {
        "domain_ages": "Snapshot Domain Ages:" in prompt,
        "fear_greed": "- Fear & Greed:" in prompt,
        "market_breadth": "- Market Breadth:" in prompt,
        "macro_sentiment": "- Macro Sentiment Advisor:" in prompt,
        "geopolitical_threat": "- Geopolitical Threat Shield:" in prompt,
        "economic_calendar": "Economic Calendar" in prompt or "macro_calendar" in prompt or "Events" in prompt,
        "macro_benchmarks": "US Equities, Futures & Macro Rates Benchmarks" in prompt,
        "etf_flows": "US Spot Crypto ETF Net Flows" in prompt,
        "live_news": "Live Breaking News Feed" in prompt,
        "vol_profile_poc": "vol_profile(POC=" in prompt,
        "session_vwap": "session_vwap=" in prompt,
        "vs_btc": "vs_btc(" in prompt,
        "kinematics": "kinematics(" in prompt,
        "kalman_vel": "kalman_vel=" in prompt,
        "no_raw_slopes": "slopes(" not in prompt and "g_coeff(" not in prompt,
        "orderbook_walls": "bid_wall=" in prompt and "ask_wall=" in prompt,
        "orderbook_depth_1pct": "depth_1pct=" in prompt,
        "cvd_15m": "cvd_15m=" in prompt,
        "spot_cvd_15m": "spot_cvd_15m=" in prompt,
        "perp_basis": "perp_basis=" in prompt,
        "funding_ann": "funding_ann=" in prompt,
        "liq_clusters": "liq_clusters(" in prompt,
        "realized_liq": "realized_liq_15m(" in prompt,
        "portfolio_risk_engine": "Portfolio Risk Engine (Crypto Scope:" in prompt,
        "free_quote_cash": "Available Free Quote Cash per Venue:" in prompt,
        "tradable_positions": "Live Tradable Spot Instruments" in prompt,
        "open_limit_orders": "open_limit_orders:" in prompt,
        "active_ai_trade_plans": "Active AI Trade Plans" in prompt or "Linked Trade Plans" in prompt,
        "decision_scorecard": "ai_decision_scorecard" in prompt or "AI Decision Scorecard" in prompt,
        "calibration_feedback": "ai_calibration_feedback" in prompt or "Calibration" in prompt,
        "cycle_cost_anchors": "Active Cycle Cost & Drawdown Anchors:" in prompt,
    }

    return {
        "prompt_length_chars": prompt_len,
        "prompt_est_tokens": est_tokens,
        "resume_prompt_length_chars": len(resume_prompt),
        "resume_prompt_est_tokens": len(resume_prompt) // 4,
        "na_count": na_count,
        "none_count": none_count,
        "nan_count": nan_count,
        "stale_count": stale_count,
        "found_symbols": found_symbols,
        "key_features": key_features,
        "sections_found": list(sec_lines.keys()),
        "full_prompt": prompt,
    }


if __name__ == "__main__":
    res = inspect_crypto_prompt(sys.argv[1] if len(sys.argv) > 1 else ".")
    full_prompt = res.pop("full_prompt")
    print(encode_json_payload(res, indent=2))
    if "--print-prompt" in sys.argv:
        print("\n" + "=" * 80)
        print("FULL PROMPT:")
        print("=" * 80)
        print(full_prompt)
