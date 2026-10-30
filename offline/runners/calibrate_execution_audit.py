#!/usr/bin/env python3
"""Generate a read-only report from the execution_audit JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cachepaths import normalize_venue_name  # noqa: E402
from state_io import atomic_write_json, atomic_write_text  # noqa: E402
from offline.backtests.comparison import format_number as _fmt  # noqa: E402
from offline.backtests.execution_calibration import (  # noqa: E402
    calibrate_execution_events,
)


def _paths(inputs: list[Path]) -> list[Path]:
    paths = []
    for path in inputs:
        resolved = path.expanduser().resolve()
        if resolved.is_dir():
            paths.extend(sorted(resolved.glob("execution_audit_*.jsonl")))
        else:
            paths.append(resolved)
    return sorted(set(paths))


def _norm_audit_venue(val: object) -> str:
    if isinstance(val, bool) or val is None:
        return ""
    raw = str(val).strip().lower()
    return normalize_venue_name(raw) or raw


def _load(paths: list[Path], venue: str | None) -> list[dict]:
    events = []
    norm_venue = _norm_audit_venue(venue)
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: JSON invalid") from error
                if not isinstance(event, dict):
                    raise ValueError(f"{path}:{line_number}: JSON invalid")
                if norm_venue and _norm_audit_venue(event.get("venue")) != norm_venue:
                    continue
                events.append(event)
    return events


def markdown_report(report: dict, paths: list[Path]) -> str:
    lines = [
        "# Execution audit calibration", "",
        f"Files: {len(paths)}", "",
        "| Tip | Orders | Filled | Partial | Fee p50 bps | Fee p95 bps | "
        "Latency p50 s | Fill ratio p50 | Shortfall p50 bps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("all", "limit", "market"):
        item = report["summary"][name]
        lines.append(
            f"| {name} | {item['orders']} | {item['filled']} | "
            f"{item['ever_partial']} | {_fmt(item['fee_bps']['p50'])} | "
            f"{_fmt(item['fee_bps']['p95'])} | "
            f"{_fmt(item['first_fill_latency_s']['p50'])} | "
            f"{_fmt(item['final_fill_ratio']['p50'])} | "
            f"{_fmt(item['market_execution_shortfall_bps']['p50'])} |"
        )
    ready = report["calibration_readiness"]
    client_ids = report["client_order_id_validation"]
    lines.extend([
        "", "## Readiness", "",
        f"Filled orders: {ready['filled_orders']}/{ready['minimum_filled_orders']} minimum.",
        "", "MARKET shortfall: available after deploy, once the audit carries "
        "`reference_price`; it is not pure slippage.",
        "", "Market slippage separat: indisponibil — "
        + ready["market_slippage_blocker"] + ".",
        "", "Spread: indisponibil — " + ready["spread_blocker"] + ".", "",
        "The report is observational and does not automatically change the backtest scenarios.", "",
        "## Client order ID", "",
        f"Orders accepted on venues with support: "
        f"{client_ids['supported_accepted_orders']}; "
        f"ID prezent: {client_ids['with_client_order_id']}; "
        f"valid: {client_ids['valid_client_order_ids']}; "
        f"invalid: {client_ids['invalid_client_order_ids']}; "
        f"missing: {client_ids['missing_client_order_ids']}.", "",
    ])
    evidence = client_ids["first_valid_by_venue"]
    if evidence:
        lines.extend([
            "| Venue | Symbol | Intent | Client order ID | Venue order ID |",
            "|---|---|---|---|---|",
        ])
        for venue, item in sorted(evidence.items()):
            lines.append(
                f"| {venue} | {item['symbol']} | {item['intent_id']} | "
                f"{item['client_order_id']} | {item['order_id']} |"
            )
        lines.extend(["", "This evidence comes from `submit_accepted`; the report places no orders.", ""])
    else:
        lines.extend([
            "PENDING: there is no `submit_accepted` with a valid ID after deploy yet; "
            "no order is placed just for this check.", "",
        ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs", nargs="+", type=Path,
        help="JSONL files or logger/execution_audit directories",
    )
    parser.add_argument("--venue")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    paths = _paths(args.inputs)
    if not paths:
        parser.error("no execution_audit files found")
    try:
        events = _load(paths, args.venue)
    except ValueError as error:
        parser.error(str(error))
    report = calibrate_execution_events(events)
    report["source_files"] = [str(path) for path in paths]
    output = args.output.expanduser().resolve() if args.output else None
    markdown = args.markdown.expanduser().resolve() if args.markdown else None
    if output or markdown:
        if output:
            atomic_write_json(output, report, indent=2, ensure_ascii=False)
        if markdown:
            atomic_write_text(markdown, markdown_report(report, paths))
    print(markdown_report(report, paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
