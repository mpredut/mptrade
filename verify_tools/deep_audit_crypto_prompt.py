#!/usr/bin/env python3
"""Deep quality audit of the auto_cryptotrade_executor prompt."""

import os
import re
import sys
from typing import Any, Dict, List

from botcore import topic_spec
from orchestratorOS.lib.technical_repair_ssot import LogEventSummary
from orchestratorOS.supervisor.autonomous_ai_reconciler import (
    AutonomousAIReconciler,
)
from orchestratorOS.supervisor.prompt_formatters import build_supervisor_prompt


def deep_audit(workspace_dir: str = ".") -> None:
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

    print("=== SUMMARY METRICS ===")
    print(f"Total Chars: {len(prompt):,}")
    print(f"Est Tokens: {len(prompt)//4:,}")
    print(f"Total Lines: {len(prompt.splitlines())}")

    print("\n=== DETAILED NA / NONE / STALE OCCURRENCES ===")
    for i, line in enumerate(prompt.splitlines(), 1):
        if any(w in line for w in ["N/A", "None", "[stale]", "STALE", "NaN", "nan"]):
            # Filter out expected strings like "None" in schema
            print(f"L{i:03d}: {line}")

    print("\n=== DOMAIN AGES (Section 0) ===")
    for line in prompt.splitlines():
        if "Domain Ages" in line or line.startswith("### 0."):
            print(line)

    print("\n=== MACRO & NEWS TELEMETRY (Section 1) ===")
    in_s1 = False
    for line in prompt.splitlines():
        if line.startswith("### 1."):
            in_s1 = True
        elif line.startswith("### 3."):
            in_s1 = False
        if in_s1:
            print(line[:160] + ("..." if len(line) > 160 else ""))

    print("\n=== SAMPLE COIN PRICE & TREND (Section 3: BTC, HYPE, ARB) ===")
    for line in prompt.splitlines():
        if any(line.strip().startswith(f"* {sym}") for sym in ["BTCUSDC", "HYPEUSDC", "ARBUSDC", "SPCXxUSD"]):
            print(line[:160] + ("..." if len(line) > 160 else ""))

    print("\n=== SAMPLE MICROSTRUCTURE (Section 3B: BTC, HYPE, ARB) ===")
    for line in prompt.splitlines():
        if any(line.strip().startswith(f"* {sym}") for sym in ["BTCUSDC", "HYPEUSDC", "ARBUSDC", "SPCXxUSD"]):
            print(line[:160] + ("..." if len(line) > 160 else ""))

    print("\n=== PORTFOLIO STATE (Section 3C) ===")
    in_s3c = False
    for line in prompt.splitlines():
        if line.startswith("### 3C."):
            in_s3c = True
        elif line.startswith("INTERACTIVE THREAD RULES:") or line.startswith("```json"):
            in_s3c = False
        if in_s3c:
            print(line[:160] + ("..." if len(line) > 160 else ""))


if __name__ == "__main__":
    deep_audit(sys.argv[1] if len(sys.argv) > 1 else ".")
