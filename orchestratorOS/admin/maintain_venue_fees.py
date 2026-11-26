#!/usr/bin/env python3
"""CLI runner for 4-hour venue fee optimization and Binance BNB fee-reserve maintenance."""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import botcore
from botcore import add_store_true_cli_args, now_str, split_csv
from notify_engine import ntfyapi
from providers.venue_fee_maintenance import run_venue_fee_maintenance
from state_io import encode_json_payload


def _dispatch_maintenance_event(ev: Dict[str, Any]) -> None:
    kind = str(ev.get("kind") or "").strip().lower()
    if kind == "order_placed":
        ntfyapi.dispatch_order_event_notification(
            ev.get("symbol"),
            ev.get("side"),
            ev.get("order_id"),
            ev.get("price"),
            ev.get("qty"),
            default_verb="PLACED",
            default_icon="",
            venue=str(ev.get("venue") or "binance"),
            owner=str(ev.get("owner") or "FEE_MAINTAINER"),
            source="fee_maintainer",
            order_type=str(ev.get("order_type") or "LIMIT_MAKER"),
            notional_usd=ev.get("notional_usd"),
            thesis=ev.get("thesis"),
        )
        return
    title = str(ev.get("title") or "").strip()
    body = str(ev.get("body") or "").strip()
    if title and body:
        ntfyapi.notify(
            title=title,
            body=body,
            source="fee_maintainer",
            symbol=str(ev.get("symbol") or "FEE"),
        )


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_store_true_cli_args(
        parser,
        ("--force", "Run immediately regardless of the 4h interval gate."),
        ("--dry-run", "Inspect venues and compute actions without mutating venue orders or disk state."),
        ("--json", "Emit the full structured result as JSON."),
    )
    parser.add_argument(
        "--venue",
        default="",
        help="Optional comma-separated subset of venues (binance,kraken,hyperliquid,t212).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    botcore.load_env_stack()
    args = build_cli_parser().parse_args(argv)
    venues = list(split_csv(args.venue, lower=True)) if args.venue else None
    result = run_venue_fee_maintenance(
        venues=venues,
        force=bool(args.force),
        dry_run=bool(args.dry_run),
        event_handler=None if args.dry_run else _dispatch_maintenance_event,
    )
    if args.json:
        print(encode_json_payload(result, indent=2, sort_keys=True))
    else:
        status = result.get("status")
        if status != "not_due" or args.force:
            bn = (result.get("venues") or {}).get("binance") or {}
            kr = (result.get("venues") or {}).get("kraken") or {}
            hl = (result.get("venues") or {}).get("hyperliquid") or {}
            print(
                f"[{now_str()}] [fee_maintainer] status={status} "
                f"binance_bnb_usdc={bn.get('held_bnb_usdc')} action={bn.get('topup_action')} "
                f"bn_tier={(bn.get('fee_spec') or {}).get('fee_tier')} "
                f"kr_bps={(kr.get('fee_spec') or {}).get('maker_fee_bps')}/{(kr.get('fee_spec') or {}).get('taker_fee_bps')} "
                f"hl_bps={(hl.get('fee_spec') or {}).get('maker_fee_bps')}/{(hl.get('fee_spec') or {}).get('taker_fee_bps')} "
                f"errors={result.get('errors') or {}}"
            )
    errors = result.get("errors") or {}
    venues_out = result.get("venues") or {}
    if errors and venues_out and len(errors) >= len(venues_out):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
