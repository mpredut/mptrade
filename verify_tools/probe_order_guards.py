"""On-demand read-only CLI probe of deterministic order guards (0 ms, zero LLM, zero ntfy).

Allows the autonomous AI executor or operator to probe what deterministic bot guards
(G_PROF, G_QLIMIT2, GI_WHALE, GI_OBOOK, GI_FUNDING, GLLM3) would decide for a specific
candidate order price before deciding whether to heed or override them.

Usage:
    ./orchestratorOS/run_python.sh verify_tools/probe_order_guards.py \
        --venue binance --symbol TAOUSDC --side BUY --price 205.0
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from botcore import disable_auto_start_websockets_default, parse_pos_cli_float
from intelligence.sentiment.evaluators.guard_probe_evaluator import probe_deterministic_guards
from state_io import encode_json_payload


def main(argv: list[str] | None = None) -> int:
    disable_auto_start_websockets_default()
    ap = argparse.ArgumentParser(description="Read-only advisory probe of deterministic MPTrade order guards.")
    ap.add_argument("--symbol", required=True, help="Instrument symbol (e.g. BTCUSDC, TAOUSDC, HYPEUSDC)")
    ap.add_argument("--side", required=True, choices=["BUY", "SELL", "buy", "sell"], help="Order side")
    ap.add_argument("--price", required=True, type=parse_pos_cli_float, help="Candidate order price")
    ap.add_argument("--venue", default="", help="Venue name (binance, kraken, hyperliquid, t212)")
    ap.add_argument(
        "--opposite-fill-px",
        type=parse_pos_cli_float,
        default=None,
        help="Optional reference opposite-side fill price for G_PROF evaluation",
    )
    args = ap.parse_args(argv)
    result = probe_deterministic_guards(
        args.symbol,
        args.side.upper(),
        args.price,
        venue=args.venue,
        opposite_fill_px=args.opposite_fill_px,
        allow_disk_cache=True,
    )
    sys.stdout.write(encode_json_payload(result, indent=2, sort_keys=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
