#!/usr/bin/env python3
"""Standalone CLI for multi-venue AI vs. BOT attribution, model-tier audit, and override evaluation.

Zero network calls, zero live trading side effects. Can run against the active workspace or any
archived backup directory.

Examples:
    ./orchestratorOS/run_python.sh verify_tools/ai_attribution_report.py --days 7
    ./orchestratorOS/run_python.sh verify_tools/ai_attribution_report.py --days 30 --scope crypto --json
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
from typing import Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

with contextlib.redirect_stdout(sys.stderr):
    from botcore import disable_auto_start_websockets_default, parse_pos_cli_float  # noqa: E402
    from providers.ai_attribution import format_ai_attribution_report  # noqa: E402
    from providers.ai_attribution_store import load_and_evaluate_ai_attribution  # noqa: E402
    from state_io import encode_json_payload, save_json_state  # noqa: E402


def main(argv: Optional[list[str]] = None) -> int:
    disable_auto_start_websockets_default()
    parser = argparse.ArgumentParser(
        description="Standalone multi-venue AI vs. BOT attribution, model_used audit, and override report"
    )
    parser.add_argument(
        "--days",
        type=parse_pos_cli_float,
        default=7.0,
        help="Lookback window in days (default: 7)",
    )
    parser.add_argument(
        "--scope",
        choices=("all", "crypto", "equities"),
        default="all",
        help="Asset scope filter (default: all)",
    )
    parser.add_argument(
        "--workspace",
        default=_ROOT,
        help="Workspace directory containing cachedb/ and logger/ (default: repository root)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit strict JSON payload to stdout",
    )
    parser.add_argument(
        "--out-json",
        default="",
        help="Optional output path to atomically persist the JSON attribution report",
    )
    args = parser.parse_args(argv)

    with contextlib.redirect_stdout(sys.stderr):
        report = load_and_evaluate_ai_attribution(
            args.workspace,
            lookback_days=args.days,
            scope=args.scope,
        )
        if args.out_json:
            save_json_state(args.out_json, report, indent=2)
    if args.as_json:
        sys.stdout.write(encode_json_payload(report, indent=2, sort_keys=True) + "\n")
    else:
        sys.stdout.write(format_ai_attribution_report(report) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
