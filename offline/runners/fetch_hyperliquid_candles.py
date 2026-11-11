#!/usr/bin/env python3
"""Freeze public Hyperliquid candles into a CSV compatible with the replay.

The HYPE/USDC spot data is a cross-venue proxy for validating the robustness of
the Kraken strategy. It does not model Kraken spread, liquidity or fills.
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HL_DIR = ROOT / "hyperliquid"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from botcore import exact_int, exact_pos_int, safe_int
from state_io import atomic_write_bytes, atomic_write_json
from offline.backtests.datasets import canonical_bytes, dataset_metadata
from offline.backtests.evaluation import (
    load_venue_module,
    parse_intervals,
    parse_optional_lookback_days,
)

_hl_md = load_venue_module("hyperliquid", "market_data.py", "local_hl_market_data")
API_URL = _hl_md.API_URL
INTERVALS = _hl_md.INTERVALS
fetch_ohlc_candles = _hl_md.fetch_ohlc_candles


def resolve_spot_pair(token: str, quote: str, spot_meta: dict) -> str:
    """Resolve the ``@index`` API name from ``spotMeta``, with no hardcoded index."""
    if not isinstance(spot_meta, dict):
        raise ValueError("spot_meta must be a dict")
    if isinstance(token, bool) or isinstance(quote, bool) or not token or not quote:
        raise ValueError(f"spot token not found: {token}/{quote}")
    norm_token = str(token).strip().upper()
    norm_quote = str(quote).strip().upper()
    raw_tokens = spot_meta.get("tokens")
    token_candidates: dict[str, list[tuple[bool, object]]] = {}
    for item in (raw_tokens if isinstance(raw_tokens, list) else []):
        if not isinstance(item, dict) or isinstance(item.get("name"), bool) or isinstance(item.get("index"), bool):
            continue
        t_name = str(item.get("name", "")).strip().upper()
        t_idx = item.get("index")
        if t_name and t_idx is not None:
            is_canon = bool(item.get("isCanonical") is True)
            token_candidates.setdefault(t_name, []).append((is_canon, t_idx))
    base_list = sorted(token_candidates.get(norm_token, ()), key=lambda pair: (not pair[0],))
    quote_list = sorted(token_candidates.get(norm_quote, ()), key=lambda pair: (not pair[0],))
    if not base_list or not quote_list:
        raise ValueError(f"spot token not found: {token}/{quote}")

    raw_universe = spot_meta.get("universe")
    universe_items = [m for m in (raw_universe if isinstance(raw_universe, list) else []) if isinstance(m, dict)]
    universe_items.sort(key=lambda m: (m.get("isCanonical") is not True,))
    for _, b_idx in base_list:
        for _, q_idx in quote_list:
            for market in universe_items:
                if market.get("tokens") == [b_idx, q_idx]:
                    name = market.get("name")
                    if name and not isinstance(name, bool):
                        return str(name).strip()
    raise ValueError(f"spot pair not found: {token}/{quote}")


def validate_continuity(records: list[dict], interval_minutes: int) -> None:
    """The replay assumes evenly spaced bars; it refuses datasets with gaps."""
    if exact_pos_int(interval_minutes, default=None) is None:
        raise ValueError("interval_minutes must be a positive integer")
    if not isinstance(records, (list, tuple)):
        raise ValueError("records must be a sequence")
    expected_seconds = interval_minutes * 60
    for previous, current in zip(records, records[1:]):
        if not isinstance(previous, dict) or not isinstance(current, dict):
            raise ValueError("discontinuous dataset: invalid record")
        prev_i = exact_int(previous.get("timestamp"), default=None)
        curr_i = exact_int(current.get("timestamp"), default=None)
        if prev_i is None or curr_i is None:
            raise ValueError("discontinuous dataset: invalid timestamp")
        actual_seconds = curr_i - prev_i
        if actual_seconds != expected_seconds:
            raise ValueError(
                "discontinuous dataset: "
                f"{previous['timestamp']} -> {current['timestamp']} "
                f"({actual_seconds}s, expected {expected_seconds}s)"
            )


def _parse_intervals(value: str) -> list[int]:
    try:
        intervals = parse_intervals(value)
    except argparse.ArgumentTypeError as exc:
        raise argparse.ArgumentTypeError("intervals must be in minutes") from exc
    unsupported = [item for item in intervals if item not in INTERVALS]
    if unsupported:
        raise argparse.ArgumentTypeError(
            f"unsupported intervals: {unsupported or value}; supported: {sorted(INTERVALS)}"
        )
    return intervals


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", default="HYPE")
    parser.add_argument("--quote", default="USDC")
    parser.add_argument("--intervals", type=_parse_intervals, default=[60, 240, 1440])
    parser.add_argument(
        "--lookback-days",
        help="limit every interval to the same period; defaults to the API maximum",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "offline/results/hyperliquid_proxy/datasets",
    )
    args = parser.parse_args(argv)
    args.lookback_days = parse_optional_lookback_days(parser, args.lookback_days)
    for label, tok in (("--token", args.token), ("--quote", args.quote)):
        if not isinstance(tok, str) or not tok.strip() or "/" in tok or "\\" in tok or ".." in tok:
            parser.error(f"invalid {label}: {tok!r}")

    generated_at = dt.datetime.now(tz=dt.timezone.utc)
    end_ms = safe_int(generated_at.timestamp() * 1000, 0)
    start_ms = (
        max(0, end_ms - args.lookback_days * 24 * 60 * 60 * 1000)
        if args.lookback_days is not None else 0
    )
    try:
        spot_meta = _hl_md._post_info({"type": "spotMeta"})
        if not isinstance(spot_meta, dict):
            raise RuntimeError("invalid spotMeta response")
        coin = resolve_spot_pair(args.token, args.quote, spot_meta)
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "schema_version": 1,
            "generated_at_utc": generated_at.isoformat(),
            "source": {
                "venue": "Hyperliquid",
                "market": f"{args.token.strip().upper()}/{args.quote.strip().upper()} spot",
                "api_coin": coin,
                "endpoint": API_URL,
                "documentation": (
                    "https://hyperliquid.gitbook.io/hyperliquid-docs/"
                    "for-developers/api/info-endpoint"
                ),
            },
            "purpose": "cross-venue robustness proxy for Kraken strategy replay",
            "limitations": [
                "not Kraken market or execution data",
                "does not reproduce Kraken spread, queue position, latency, or fills",
                "Hyperliquid exposes at most the most recent 5000 candles per market",
            ],
            "requested_lookback_days": args.lookback_days,
            "datasets": {},
        }
        stem = f"{args.token.strip().upper()}{args.quote.strip().upper()}"
        for interval in args.intervals:
            records = fetch_ohlc_candles(
                coin, interval, start_ms=start_ms, end_ms=end_ms, drop_incomplete=True,
            )
            if not records:
                raise RuntimeError(f"an empty dataset for {coin} {interval}m")
            validate_continuity(records, interval)
            raw = canonical_bytes(records)
            meta = dataset_metadata(records, interval_minutes=interval)
            digest = meta["sha256"]
            path = output_dir / f"{stem}_{interval}m_hlspot_{digest[:12]}.csv"
            atomic_write_bytes(path, raw)
            manifest["datasets"][str(interval)] = {
                "file": str(path),
                **meta,
            }
            print(f"{interval}m: {len(records)} bars -> {path}")

        stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
        manifest_path = output_dir.parent / f"manifest_{stamp}.json"
        atomic_write_json(manifest_path, manifest, indent=2, ensure_ascii=True)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
