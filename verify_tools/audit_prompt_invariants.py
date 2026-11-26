"""CLI runner for deterministic mathematical and structural invariant verification of AI executor telemetry & prompts.

Delegates all pure invariant logic to ``intelligence.sentiment.evaluators.prompt_invariants_evaluator``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from botcore import load_key_value_conf, resolve_now_ts, topic_spec
from instrument_registry import disabled_registry_symbols, observe_only_registry_symbols
from intelligence.sentiment.evaluators.prompt_invariants_evaluator import (
    InvariantViolation,
    audit_rendered_prompt_text,
    audit_scorecard_temporal_invariants,
    audit_snapshot_invariants,
)
from providers.intent_state import parse_intent_gateway_config
from state_io import read_text_file


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit AI executor telemetry snapshot or rendered prompt invariants.")
    parser.add_argument("--workspace-dir", default=".", help="Workspace directory containing instruments.conf")
    parser.add_argument("--cache-dir", default="cachedb", help="Cache directory")
    parser.add_argument("--prompt-file", default="", help="Path to a rendered prompt text file to audit")
    parser.add_argument("--live-snapshot", action="store_true", help="Collect an offline snapshot from cachedb and audit it")
    parser.add_argument("--check-rendered-prompt", action="store_true", help="Also render full and compact prompts from --live-snapshot and audit them")
    parser.add_argument("--check-scorecard-stability", action="store_true", help="Verify scorecard LIFO stability on durable intents")
    parser.add_argument("--scope", default="crypto", help="Asset scope for live snapshot / scorecard audit")
    args = parser.parse_args(argv)

    non_trad = disabled_registry_symbols(args.workspace_dir) | observe_only_registry_symbols(args.workspace_dir)
    try:
        conf = load_key_value_conf(args.workspace_dir, str(_ROOT))
        _, gw_min_usd, gw_max_usd, gw_min_conf, _ = parse_intent_gateway_config(conf)
    except Exception:
        gw_min_usd, gw_max_usd, gw_min_conf = 10.0, 2000.0, 0.85
    violations: list[InvariantViolation] = []

    if args.prompt_file:
        text = read_text_file(args.prompt_file)
        violations.extend(
            audit_rendered_prompt_text(text, non_tradable_symbols=non_trad, gateway_min_usd=gw_min_usd)
        )

    if args.live_snapshot:
        from intelligence.sentiment.collectors.macro_telemetry_collector import MarketAndPortfolioSnapshotOrchestrator

        snap = MarketAndPortfolioSnapshotOrchestrator().collect(
            args.cache_dir,
            "*",
            workspace_dir=args.workspace_dir,
            scope=args.scope,
            allow_network=False,
            persist=False,
        )
        violations.extend(
            audit_snapshot_invariants(
                snap, non_tradable_symbols=non_trad, gateway_min_usd=gw_min_usd
            )
        )
        if args.check_rendered_prompt:
            from orchestratorOS.supervisor.prompt_formatters import build_supervisor_prompt

            scope_l = str(args.scope or "crypto").strip().lower()
            tkey = "auto_sharetrade_executor" if scope_l in ("equities", "equity") else "auto_cryptotrade_executor"
            spec = topic_spec(tkey)
            for compact_flag in (False, True):
                rendered = build_supervisor_prompt(
                    snap,
                    logs=None,
                    spec=spec,
                    min_usd=gw_min_usd,
                    max_usd=gw_max_usd,
                    min_conf=gw_min_conf,
                    compact_rules=compact_flag,
                )
                violations.extend(
                    audit_rendered_prompt_text(
                        rendered, non_tradable_symbols=non_trad, gateway_min_usd=gw_min_usd
                    )
                )

    if args.check_scorecard_stability:
        from providers.intent_state_store import load_durable_intents_from_cachedb

        intents = load_durable_intents_from_cachedb(args.cache_dir)
        now_s = resolve_now_ts(None)
        violations.extend(
            audit_scorecard_temporal_invariants(
                intents,
                t_early=now_s - 3600.0,
                t_late=now_s,
                scope=args.scope,
            )
        )

    if violations:
        for v in violations:
            print(f"FAIL {v.format_line()}", file=sys.stderr)
        print(f"Audit failed with {len(violations)} invariant violation(s).", file=sys.stderr)
        return 1

    print("OK: 0 prompt/telemetry invariant violations detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
