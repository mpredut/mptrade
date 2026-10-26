#!/usr/bin/env python3
"""Freeze public Hyperliquid candles into a CSV compatible with the replay.

The HYPE/USDC spot data is a cross-venue proxy for validating the robustness of
the Kraken strategy. It does not model Kraken spread, liquidity or fills.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
HL_DIR = ROOT / "hyperliquid"
sys.path.insert(0, str(HL_DIR))
sys.path.insert(0, str(ROOT))

import importlib.util
_spec = importlib.util.spec_from_file_location("local_hl_market_data", str(HL_DIR / "market_data.py"))
_hl_md = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hl_md)
API_URL = _hl_md.API_URL
INTERVALS = _hl_md.INTERVALS
fetch_ohlc_candles = _hl_md.fetch_ohlc_candles
normalize_closed_candles = _hl_md.normalize_closed_candles

from offline.backtests.datasets import canonical_bytes, dataset_metadata, dataset_sha256


_post = _hl_md._post_info


def resolve_spot_pair(token: str, quote: str, spot_meta: dict) -> str:
    """Resolve the ``@index`` API name from ``spotMeta``, with no hardcoded index."""
    if not isinstance(spot_meta, dict):
        raise ValueError("spot_meta must be a dict")
    raw_tokens = spot_meta.get("tokens")
    token_indices = {
        str(item.get("name", "")).upper(): item.get("index")
        for item in (raw_tokens if isinstance(raw_tokens, list) else [])
        if isinstance(item, dict)
    }
    base_index = token_indices.get(str(token or "").upper())
    quote_index = token_indices.get(str(quote or "").upper())
    if base_index is None or quote_index is None:
        raise ValueError(f"spot token not found: {token}/{quote}")
    raw_universe = spot_meta.get("universe")
    for market in (raw_universe if isinstance(raw_universe, list) else []):
        if isinstance(market, dict) and market.get("tokens") == [base_index, quote_index]:
            name = market.get("name")
            if name:
                return str(name)
    raise ValueError(f"spot pair not found: {token}/{quote}")


def validate_continuity(records: list[dict], interval_minutes: int) -> None:
    """The replay assumes evenly spaced bars; it refuses datasets with gaps."""
    if isinstance(interval_minutes, bool) or not isinstance(interval_minutes, int) or interval_minutes <= 0:
        raise ValueError("interval_minutes must be a positive integer")
    expected_seconds = interval_minutes * 60
    for previous, current in zip(records, records[1:]):
        actual_seconds = current["timestamp"] - previous["timestamp"]
        if actual_seconds != expected_seconds:
            raise ValueError(
                "discontinuous dataset: "
                f"{previous['timestamp']} -> {current['timestamp']} "
                f"({actual_seconds}s, expected {expected_seconds}s)"
            )


def fetch_closed_candles(
    coin: str,
    interval_minutes: int,
    *,
    start_ms: int,
    end_ms: int,
) -> list[dict]:
    return fetch_ohlc_candles(
        coin,
        interval_minutes,
        start_ms=start_ms,
        end_ms=end_ms,
        drop_incomplete=True,
    )


def _parse_intervals(value: str) -> list[int]:
    try:
        intervals = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("intervals must be in minutes") from exc
    unsupported = [value for value in intervals if value not in INTERVALS]
    if not intervals or unsupported:
        raise argparse.ArgumentTypeError(
            f"intervale nesuportate: {unsupported or value}; suportate: {sorted(INTERVALS)}"
        )
    return intervals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", default="HYPE")
    parser.add_argument("--quote", default="USDC")
    parser.add_argument("--intervals", type=_parse_intervals, default=[60, 240, 1440])
    parser.add_argument(
        "--lookback-days", type=int,
        help="limit every interval to the same period; defaults to the API maximum",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("offline/results/hyperliquid_proxy/datasets"),
    )
    args = parser.parse_args()
    if args.lookback_days is not None and args.lookback_days <= 0:
        parser.error("--lookback-days must be positive")

    generated_at = dt.datetime.now(tz=dt.timezone.utc)
    end_ms = int(generated_at.timestamp() * 1000)
    start_ms = (
        end_ms - args.lookback_days * 24 * 60 * 60 * 1000
        if args.lookback_days is not None else 0
    )
    spot_meta = _post({"type": "spotMeta"})
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
            "market": f"{args.token.upper()}/{args.quote.upper()} spot",
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
    stem = f"{args.token.upper()}{args.quote.upper()}"
    for interval in args.intervals:
        records = fetch_closed_candles(
            coin, interval, start_ms=start_ms, end_ms=end_ms,
        )
        if not records:
            raise RuntimeError(f"an empty dataset for {coin} {interval}m")
        validate_continuity(records, interval)
        raw = canonical_bytes(records)
        meta = dataset_metadata(records, interval_minutes=interval)
        digest = meta["sha256"]
        path = output_dir / f"{stem}_{interval}m_hlspot_{digest[:12]}.csv"
        path.write_bytes(raw)
        manifest["datasets"][str(interval)] = {
            "file": str(path),
            **meta,
        }
        print(f"{interval}m: {len(records)} bare -> {path}")

    stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    manifest_path = output_dir.parent / f"manifest_{stamp}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
