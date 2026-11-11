#!/usr/bin/env python3
"""Generate a read-only report from the execution_audit JSONL files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cachepaths import normalize_venue_name  # noqa: E402
from state_io import atomic_write_json, atomic_write_text, iter_jsonl_dicts  # noqa: E402
from offline.backtests.comparison import format_number  # noqa: E402
from offline.backtests.execution_calibration import (  # noqa: E402
    calibrate_execution_events,
)


def _paths(inputs: list[Path]) -> list[Path]:
    paths = []
    for path in inputs:
        resolved = path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"input path does not exist: {resolved}")
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


def _is_missing_audit_venue(val: object) -> bool:
    return val is None or (isinstance(val, str) and not val.strip())


def _clean_intent_id(val: object) -> str:
    return val.strip() if isinstance(val, str) else ""


def _load(paths: list[Path], venue: str | None) -> list[dict]:
    parsed_events: list[dict] = []
    norm_venue = _norm_audit_venue(venue)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"execution audit file not found: {path}")
        try:
            for event in iter_jsonl_dicts(str(path), strict=True):
                parsed_events.append(event)
        except ValueError as exc:
            raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not norm_venue:
        return parsed_events
    matched_intents = {
        intent_id
        for event in parsed_events
        if (intent_id := _clean_intent_id(event.get("intent_id")))
        and _norm_audit_venue(event.get("venue")) == norm_venue
    }
    return [
        event
        for event in parsed_events
        if _norm_audit_venue(event.get("venue")) == norm_venue
        or (
            _is_missing_audit_venue(event.get("venue"))
            and _clean_intent_id(event.get("intent_id")) in matched_intents
        )
    ]


def markdown_report(report: dict, paths: list[Path]) -> str:
    lines = [
        "# Execution audit calibration", "",
        f"Files: {len(paths)}", "",
        "| Type | Orders | Filled | Partial | Fee p50 bps | Fee p95 bps | "
        "Latency p50 s | Fill ratio p50 | Shortfall p50 bps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("all", "limit", "market"):
        item = report["summary"][name]
        lines.append(
            f"| {name} | {item['orders']} | {item['filled']} | "
            f"{item['ever_partial']} | {format_number(item['fee_bps']['p50'])} | "
            f"{format_number(item['fee_bps']['p95'])} | "
            f"{format_number(item['first_fill_latency_s']['p50'])} | "
            f"{format_number(item['final_fill_ratio']['p50'])} | "
            f"{format_number(item['market_execution_shortfall_bps']['p50'])} |"
        )
    ready = report["calibration_readiness"]
    client_ids = report["client_order_id_validation"]
    lines.extend([
        "", "## Readiness", "",
        f"Filled orders: {ready['filled_orders']}/{ready['minimum_filled_orders']} minimum.",
        "", "MARKET shortfall: available after deploy, once the audit carries "
        "`reference_price`; it is not pure slippage.",
        "", "Separate market slippage: unavailable -- "
        + ready["market_slippage_blocker"] + ".",
        "", "Spread: unavailable -- " + ready["spread_blocker"] + ".", "",
        "The report is observational and does not automatically change the backtest scenarios.", "",
        "## Client order ID", "",
        f"Orders accepted on venues with support: "
        f"{client_ids['supported_accepted_orders']}; "
        f"ID present: {client_ids['with_client_order_id']}; "
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs", nargs="+", type=Path,
        help="JSONL files or logger/execution_audit directories",
    )
    parser.add_argument("--venue")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)
    try:
        paths = _paths(args.inputs)
        if not paths:
            parser.error("no execution_audit files found")
        events = _load(paths, args.venue)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    report = calibrate_execution_events(events)
    report["source_files"] = [str(path) for path in paths]
    md_text = markdown_report(report, paths)
    try:
        output = args.output.expanduser().resolve() if args.output else None
        markdown = args.markdown.expanduser().resolve() if args.markdown else None
        if output:
            atomic_write_json(output, report, indent=2, ensure_ascii=True)
        if markdown:
            atomic_write_text(markdown, md_text)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(md_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
