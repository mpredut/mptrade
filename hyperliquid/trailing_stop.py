#!/usr/bin/env python3
"""
trailing_stop.py (Hyperliquid) -- crash circuit breaker for Hyperliquid spot holdings.

Thin Hyperliquid venue entrypoint backed by the shared
``trailing_core.StrategyExecutorTrailingAdapter`` and ``TrailingCore`` state machine.
Configuration lives in ``instruments.conf`` (provider=hyperliquid, role.trailing=yes)
and ``hyperliquid/trailing.conf``.

  python3 trailing_stop.py          # loop (enabled/dry-run in trailing.conf)
  python3 trailing_stop.py --once   # one check
  python3 trailing_stop.py --status # current peaks and thresholds
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.append(_HERE)

from botcore import (  # noqa: E402
    clean_upper_scalar,
    load_dotenv,
    load_env_stack,
    load_local_module,
    log,
    optional_env,
    required_bool_env,
    required_float_env,
    required_int_env,
)
from cachepaths import (  # noqa: E402
    CACHE_SHORT_TREND_FILENAME,
    cache_path,
    canonical_base_asset,
    resolve_venue_trailing_state_path,
)
from providers.execution_audit import AuditedStrategyExecutor  # noqa: E402
from providers.hyperliquid_provider import HyperliquidProvider  # noqa: E402
from trailing_core import (  # noqa: E402
    StrategyExecutorTrailingAdapter,
    load_trailing_common_env,
    load_venue_trailing_registry,
    parse_trailing_cli_args,
)

_common_mod = load_local_module(_HERE, "common")
notify = _common_mod.notify

STATE_FILE = resolve_venue_trailing_state_path("hyperliquid")
CACHE_TREND = cache_path(CACHE_SHORT_TREND_FILENAME)

# Match launch order used by Hyperliquid bots: real env, secrets, versioned policy,
# then dedicated trailing policy layer.
load_env_stack(os.path.join(_HERE, ".env"))
load_dotenv(os.path.join(_HERE, "trailing.conf"))

DEFAULT_TRAIL_PCT = required_float_env("HL_TRAILING_DEFAULT_PCT")
MIN_NOTIONAL_USD = required_float_env("HL_TRAILING_MIN_NOTIONAL_USD")
(
    CHECK_SECONDS,
    REBUY_ENABLED,
    REBUY_BOUNCE_PCT,
    REBUY_SKIP_IF_TREND_DOWN,
    SELL_SKIP_IF_TREND_UP,
    SELL_TREND_DEFER_MAX_SEC,
    MIN_PROFIT_PCT,
) = load_trailing_common_env("HL_TRAILING_")
TRAILING_MARKET = required_bool_env("HL_TRAILING_MARKET")
REBUY_TREND_DAYS = required_int_env("HL_TRAILING_REBUY_TREND_DAYS")
if not 1 <= REBUY_TREND_DAYS < 1000:
    raise ValueError("HL_TRAILING_REBUY_TREND_DAYS must be in [1, 999]")

(
    TRAILING_INSTRUMENTS,
    TRAIL_PCT,
    PAIR_FOR,
    REBUY_MODE_BY_ASSET,
    QUOTE_FOR,
) = load_venue_trailing_registry(
    "hyperliquid",
    env_prefix="HL_TRAILING_",
    key_by_asset=True,
    default_trail_pct=DEFAULT_TRAIL_PCT,
)


def _default_primary_token() -> str:
    if TRAIL_PCT:
        return next(iter(TRAIL_PCT))
    for env_key in ("HL_SPOT_TOKEN", "HL_COIN"):
        raw = clean_upper_scalar(optional_env(env_key, ""))
        if raw:
            return raw
    return "HYPE"


class HyperliquidTrailing(StrategyExecutorTrailingAdapter):
    """Thin Hyperliquid spot adapter delegating all trailing logic to ``StrategyExecutorTrailingAdapter``."""

    _VENUE_SLUG = "hyperliquid"
    _VENUE_LABEL = "Hyperliquid"
    _HEARTBEAT_NAME = "HL-trailing"
    _LOG_TAG = "[TRAIL-HL]"
    _ENABLED_ENV_VAR = "HL_TRAILING_ENABLED"

    def __init__(
        self,
        client=None,
        *,
        provider=None,
        log=log,
        enabled=None,
        state_file=STATE_FILE,
        min_profit_pct=MIN_PROFIT_PCT,
        trailing_market: bool | None = None,
        rebuy_trend_days: int | None = REBUY_TREND_DAYS,
    ):
        self.client = client
        self._explicit_provider = provider is not None
        primary_tok = _default_primary_token()
        primary_provider = provider if provider is not None else HyperliquidProvider(token=primary_tok, client=client)
        self._init_executor_adapter(
            primary_provider,
            log=log,
            enabled=enabled,
            state_file=state_file,
            min_profit_pct=min_profit_pct,
            trailing_market=trailing_market,
            notify_fn=notify,
            rebuy_trend_days=rebuy_trend_days,
        )
        self._executors_by_asset[primary_tok] = self.executor

    def _executor_for(self, pair: str = ""):
        if self._explicit_provider or not pair:
            return self.executor
        base = canonical_base_asset(pair) or clean_upper_scalar(pair) or ""
        if not base:
            return self.executor
        cached = self._executors_by_asset.get(base)
        if cached is not None:
            return cached
        exec_api = AuditedStrategyExecutor(
            HyperliquidProvider(token=base, client=self.client),
            venue=self._VENUE_LABEL,
        )
        self._executors_by_asset[base] = exec_api
        return exec_api


def main() -> int:
    args = parse_trailing_cli_args(
        "A trailing-stop circuit breaker on Hyperliquid spot (with a re-buy).",
        "hyperliquid_trailing",
    )
    ts = HyperliquidTrailing()
    return ts.dispatch_cli(args)


if __name__ == "__main__":
    sys.exit(main())
