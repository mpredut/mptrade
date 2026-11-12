#!/usr/bin/env python3
"""Forward shadow for HYPE spot on Hyperliquid without order access.

Use the faithful ``strategies.spot_engine`` engine through shared replay and public
Hyperliquid candles. Never read or write live state, orders, or balances. Variants
are preregistered and remain PAPER:

* ``current``: effective HLC configuration;
* ``tp_regime_gate``: common-regime gate for newly armed TP trailing;
* ``long_tp3_trail3``: TP armed at 3%, trend hold, fixed 3% trailing;
* ``reentry4``: current configuration with reentry after a 4% pullback;
* ``trail_profit_floor_sl18``: trailing only above +1%, hard stop at -18%;
* ``overlay650t8_regime_v2``: trend overlay with 650 top-up and 8% trailing;
* ``B_dcabrake_regime_v2``: common-regime DCA brake, kept paper-only.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KRAKEN_DIR = os.path.join(ROOT, "kraken")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from botcore import (  # noqa: E402
    add_store_true_cli_args,
    load_local_module,
    load_shadow_environment,
    optional_env,
    prepare_shadow_runtime,
    require_shadow_interval,
    required_env,
    safe_int,
)
from providers.ohlc_utils import validate_ohlc_candle_bar  # noqa: E402
from strategies.spot_engine import StratParams  # noqa: E402
prepare_shadow_runtime(ROOT, HERE, KRAKEN_DIR)
from kraken import shadow_live as shadow  # noqa: E402
_common_mod = load_local_module(HERE, "common")
load_isolated_hl_cfg = _common_mod.load_isolated_hl_cfg


LOG_DIR = os.path.join(ROOT, "logs", "hyperliquid_shadow")


def _variants(interval: int):
    require_shadow_interval(interval, 240, "shadow_longterm")
    cfg = load_isolated_hl_cfg(HERE)

    base = StratParams.from_env(cfg)
    variants = {
        "current": base,
        "tp_regime_gate": dataclasses.replace(
            base, tp_regime_gate=True,
        ),
        "long_tp3_trail3": dataclasses.replace(
            base,
            takeprofit_pct=3.0,
            tp_trend_hold=True,
            tp_trail_adaptive=False,
            tp_trail_pct=3.0,
        ),
        "reentry4": dataclasses.replace(base, reentry_drop_pct=4.0),
        "trail_profit_floor_sl18": dataclasses.replace(
            base,
            tp_trail_profit_floor_pct=1.0,
            stop_loss_pct=18.0,
        ),
        "overlay650t8_regime_v2": dataclasses.replace(
            base,
            trend_overlay=True,
            trend_topup=650.0,
            trend_trail_pct=8.0,
            trend_exit_break=False,
        ),
        "B_dcabrake_regime_v2": dataclasses.replace(
            base,
            dca_trend_brake=True,
            dca_brake_min_pct=1.5,
        ),
    }
    return variants


def _fetch_with_ts(_pair: str, interval: int):
    require_shadow_interval(interval, 240, "shadow_longterm candles")
    hl_mod = sys.modules.get("hl_client")
    if hl_mod is None or not hasattr(hl_mod, "HLClient"):
        hl_mod = load_local_module(HERE, "hl_client", module_key="hl_client")
    HLClient = hl_mod.HLClient

    cfg = load_isolated_hl_cfg(HERE)
    load_shadow_environment(os.path.join(HERE, ".env"))
    token = (optional_env("HL_SPOT_TOKEN") or cfg.get("HL_SPOT_TOKEN") or "").strip()
    if not token:
        token = required_env("HL_SPOT_TOKEN")
    spot_pair = (optional_env("HL_SPOT_PAIR") or cfg.get("HL_SPOT_PAIR") or "").strip()
    client = HLClient.public()  # the unsigned client cannot submit orders
    if not spot_pair and not token.startswith("@"):
        resolve_fn = getattr(client, "resolve_spot_pair", None)
        if callable(resolve_fn):
            resolved = resolve_fn(token)
            if isinstance(resolved, str) and resolved.strip():
                spot_pair = resolved.strip()
    candle_coin = spot_pair or token
    candles = client.candles(candle_coin, "4h", lookback_hours=5000 * 4)
    now_ms = safe_int(time.time() * 1000, default=0)
    has_close_ts = False
    parsed: list[tuple[int, float, float, float, float]] = []
    for item in candles or []:
        if not isinstance(item, dict):
            continue
        raw_close_ts = item.get("T")
        if raw_close_ts is not None:
            has_close_ts = True
            bar = validate_ohlc_candle_bar(
                item.get("t"),
                item.get("o"),
                item.get("h"),
                item.get("l"),
                item.get("c"),
                close_ts=raw_close_ts,
                now_ms=now_ms,
                drop_incomplete=True,
                ts_in_ms=True,
                close_inclusive=True,
            )
        else:
            bar = validate_ohlc_candle_bar(
                item.get("t"),
                item.get("o"),
                item.get("h"),
                item.get("l"),
                item.get("c"),
                drop_incomplete=False,
                ts_in_ms=True,
            )
        if bar is not None:
            parsed.append((bar["timestamp"], bar["open"], bar["high"], bar["low"], bar["close"]))
    by_ts = {r[0]: r for r in parsed}
    rows = [by_ts[k] for k in sorted(by_ts)]
    # When candle close timestamps ('T') are absent in mock/legacy payloads, drop the final bar.
    return rows if has_close_ts else rows[:-1]


def snapshot(*, quiet: bool = False):
    prev_log_dir = shadow.LOG_DIR
    prev_variants = shadow._variants
    prev_fetch = shadow._fetch_with_ts
    shadow.LOG_DIR = LOG_DIR
    shadow._variants = _variants
    shadow._fetch_with_ts = _fetch_with_ts
    try:
        return shadow.snapshot("HYPE-HL", 240, 0.07, quiet=quiet)
    finally:
        shadow.LOG_DIR = prev_log_dir
        shadow._variants = prev_variants
        shadow._fetch_with_ts = prev_fetch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_store_true_cli_args(parser, "--quiet")
    args = parser.parse_args(argv)
    load_shadow_environment(os.path.join(HERE, ".env"))
    try:
        snapshot(quiet=args.quiet)
    except Exception as exc:  # visible cron failure without live effects
        print(f"[hl_shadow] error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
