#!/usr/bin/env python3
"""
trailing_stop.py — per-coin trailing stop for Binance holdings.

Why it exists: the historical AssetGuardian growth exit passed through
place_safe_order(force=False) and apply_weight_limit. During an uptrend the
counter-trend weight could reduce the order to zero. The trailing stop instead:
  * holds the position while price rises and tracks the peak;
  * sells only when price falls trail% from the peak, protecting realized gains;
  * uses force=True to bypass weighting, which would otherwise zero the order.

Walk-forward out-of-sample results on a 291-day real feed show that a TIGHT
trailing stop does not beat holding: rebounds during declines create whipsaw and
fees. It is not a profit source. Its intended role is a CRASH CIRCUIT BREAKER with
a WIDE threshold (~22%), triggered only by a sustained collapse as protection
against the scenario that destroys the holding. Run it in dry-run first; the rest
of the strategy (hold+DCA+weighting) remains unchanged.

The trailing and re-buy state machine is in trailing_core.TrailingCore and shared
with Kraken. This module is only the Binance ADAPTER (API and provider-specific
logging). See tests/test_trailing_stop.py for behavior coverage.

SAFETY:
  * TRAILING_ENABLED=false (default) enables DRY-RUN and only logs proposed sales.
  * Only enabled instruments with role.trailing in instruments.conf are managed.
  * The peak is persisted across restarts and is not reset.
  * Orders below the minimum notional are skipped.

  TRAILING_ENABLED=true python trailing_stop.py            # loop
  python trailing_stop.py --once                            # one check (dry-run)
  python trailing_stop.py --status                          # current peaks and thresholds
"""

from __future__ import annotations

import math
import os
import statistics
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # binance_api/ -> repository root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)   # also support direct execution (python binance_api/trailing_stop.py)

import instrument_registry as sym
from instrument_registry import select_instruments
from order_lifecycle import (
    TrackedOrderLifecycle,
    propagate_submission_refusal,
)
from providers.market_api import api as po   # single guarded .place() proxy; formerly bapi_placeorder

from trailing_core import (  # noqa: E402
    COST_BASIS_LOOKBACK_S,
    TrailingAdapterBase,
    load_trailing_common_env,
    parse_trailing_cli_args,
)
from botcore import (clean_upper_token, load_dotenv, log, required_bool_env, required_float_env,
                     required_int_env, safe_float, safe_nonneg_float, safe_pos_float)  # noqa: E402
from cachepaths import resolve_venue_trailing_state_path  # noqa: E402

DEFAULT_STATE = resolve_venue_trailing_state_path("binance", resolve_subdir=True)


# Load KEY=VALUE configuration from trailing.conf into the environment; external values override it.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "trailing.conf"))

# A WIDE THRESHOLD is a CRASH CIRCUIT BREAKER, not a profit tool.
# Walk-forward out-of-sample analysis on a real 291-day feed showed that tight
# trailing stops (8-12%) do not beat holding because violent rebounds cause whipsaw
# and fees. The useful role is protection against a sustained collapse: a wide
# threshold (~22%) triggers only on a catastrophic fall, not market noise.
# Membership and thresholds are explicit; adding market data alone never arms a
# trailing exit. Historical peaks and warm-up remain exclusively in runtime state.
TRAILING_INSTRUMENTS = {
    spec.symbol: spec for spec in select_instruments("binance", "trailing").values()
}
TRAIL_PCT = {symbol: spec.number("trailing.pct")
             for symbol, spec in TRAILING_INSTRUMENTS.items()}
# Per-coin re-buy switch (registry trailing.rebuy), replacing the single global.
# Re-buy after a stop rides continuations in an uptrend but bleeds in a downtrend
# (backtested), so it is now decided per instrument.
REBUY_MODE_BY_SYMBOL = {symbol: spec.rebuy_mode()
                        for symbol, spec in TRAILING_INSTRUMENTS.items()}
TRAILING_ENABLED = required_bool_env("TRAILING_ENABLED")
SELL_FRACTION = required_float_env("TRAILING_SELL_FRACTION")
MIN_NOTIONAL_USD = 11.0
(
    CHECK_SECONDS,
    REBUY_ENABLED,
    REBUY_BOUNCE_PCT,
    REBUY_SKIP_IF_TREND_DOWN,
    SELL_SKIP_IF_TREND_UP,
    SELL_TREND_DEFER_MAX_SEC,
    MIN_PROFIT_PCT,
) = load_trailing_common_env("TRAILING_")
REBUY_TRANCHES = required_int_env("TRAILING_REBUY_TRANCHES")
# `trailing.rebuy = auto` follows a long-term trend: re-buy only while price is above
# its N-day SMA. N = TRAILING_REBUY_TREND_DAYS (calendar days of DAILY closes). The
# closed-candle average is cached per UTC day; the live comparison is never cached.
REBUY_TREND_DAYS = required_int_env("TRAILING_REBUY_TREND_DAYS")
if not 1 <= REBUY_TREND_DAYS < 1000:
    raise ValueError("TRAILING_REBUY_TREND_DAYS must be in [1, 999] for the kline request")


def _finite(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    parsed = safe_float(value, default=None)
    if parsed is None:
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return parsed


class TrailingStop(TrailingAdapterBase):
    """Binance adapter for TrailingCore, providing balance, price, sell/buy,
    trend APIs, and provider-specific logging. TrailingCore owns the state machine."""

    _VENUE_SLUG = "binance"
    _VENUE_LABEL = "Binance"
    _HEARTBEAT_NAME = "Binance-trailing"
    _LOG_TAG = "[TRAIL]"
    _ENABLED_ENV_VAR = "TRAILING_ENABLED"

    def __init__(self, api, po, sym, log=log, enabled=None,
                 sell_fraction=SELL_FRACTION, state_file=DEFAULT_STATE,
                 min_profit_pct=MIN_PROFIT_PCT):
        self.api = api
        self.po = po
        self.sym = sym
        self.sell_fraction = _finite(
            sell_fraction, name="TRAILING_SELL_FRACTION", minimum=0.0, maximum=1.0)
        min_profit_pct = _finite(
            min_profit_pct, name="TRAILING_MIN_PROFIT_PCT", minimum=0.0)
        _finite(CHECK_SECONDS, name="TRAILING_CHECK_SECONDS", minimum=0.1)
        _finite(REBUY_BOUNCE_PCT, name="TRAILING_REBUY_BOUNCE_PCT", minimum=0.0)
        if REBUY_TRANCHES != 1:
            raise ValueError("TRAILING_REBUY_TRANCHES currently supports only the value 1")
        self._balances = []
        self._long_trend_cache = {}   # symbol -> (SMA, UTC day) for auto re-buy
        self._init_adapter_core(
            log=log,
            enabled=enabled,
            default_enabled=TRAILING_ENABLED if enabled is None else False,
            state_file=state_file,
            min_notional=MIN_NOTIONAL_USD,
            rebuy_enabled=REBUY_ENABLED,
            rebuy_bounce_pct=REBUY_BOUNCE_PCT,
            rebuy_skip_if_trend_down=REBUY_SKIP_IF_TREND_DOWN,
            sell_skip_if_trend_up=SELL_SKIP_IF_TREND_UP,
            sell_trend_defer_max_sec=SELL_TREND_DEFER_MAX_SEC,
            sell_fraction=self.sell_fraction,
            item_isolation=True,
            min_profit_pct=min_profit_pct,
        )
        self.order_lifecycle = TrackedOrderLifecycle(
            self.po, provider_name="BINANCE", venue="Binance",
            missing_confirmations=1, retry_on_lookup_error=True,
            max_age_seconds=180,
        )

    def _long_trend_up(self, symbol: str) -> bool:
        """Compare a fresh price with a complete, contiguous closed-day SMA.

        Cache only the average, never its price-dependent verdict. Refresh after
        UTC midnight; missing/stale/malformed history or price blocks auto re-buy.
        """
        symbol = clean_upper_token(symbol)
        if symbol is None:
            return False
        now = time.time()
        day = math.floor(now / 86400.0)
        cached = self._long_trend_cache.get(symbol)
        try:
            if cached and cached[1] == day:
                sma = cached[0]
            else:
                klines = self.api.client.get_klines(
                    symbol=symbol, interval="1d", limit=REBUY_TREND_DAYS + 1) or []
                closed = [
                    k for k in klines
                    if isinstance(k, (list, tuple)) and len(k) >= 7
                    and not any(isinstance(k[i], bool) for i in (0, 4, 6))
                    and (safe_float(k[6], default=None) is not None and safe_float(k[6], default=0.0) < day * 86400000)
                ]
                window = closed[-REBUY_TREND_DAYS:]
                if len(window) != REBUY_TREND_DAYS:
                    raise ValueError("insufficient completed daily history")
                closes = []
                for index, kline in enumerate(window):
                    expected_open = (day - REBUY_TREND_DAYS + index) * 86400000
                    close = safe_pos_float(kline[4], default=None)
                    open_t = safe_float(kline[0], default=None)
                    close_t = safe_float(kline[6], default=None)
                    if (open_t != expected_open
                            or close_t != expected_open + 86400000 - 1
                            or close is None):
                        raise ValueError("stale, discontinuous or invalid daily history")
                    closes.append(close)
                sma = safe_pos_float(statistics.fmean(closes), default=None)
                if sma is None:
                    raise ValueError("invalid daily average")
                self._long_trend_cache[symbol] = (sma, day)
            price = _finite(self.api.get_current_price(symbol), name="re-buy price", minimum=0)
            return price > 0 and price > sma
        except Exception as e:  # noqa: BLE001
            self.log(f"  [TRAIL] long-trend({symbol}) unavailable ({e}) — auto rebuy OFF")
            return False

    @staticmethod
    def _find_balance_row(balances: list | None, asset: str, *, unique: bool = False) -> dict | None:
        if (target := clean_upper_token(asset)) is None:
            return None
        matches = [
            row for row in (balances or [])
            if isinstance(row, dict)
            and clean_upper_token(row.get("asset")) == target
        ]
        if not matches or (unique and len(matches) != 1):
            return None
        return matches[0]

    def cost_basis(self, pair: str):
        """Use reconciled BUY/SELL inventory for new-position warm-up, when known.

        Include locked quantity when matching the account to fills. On missing,
        stale, or inconsistent evidence, retain the core's first-observation fallback.
        Existing peaks, warm-up thresholds, and pending orders are never rewritten.
        """
        if (pair_norm := clean_upper_token(pair)) is None:
            return None
        try:
            spec = self._lookup_symbol_entry(TRAILING_INSTRUMENTS, pair) or TRAILING_INSTRUMENTS[pair]
            asset = spec.base
            balance = self._find_balance_row(self._balances, asset, unique=True)
            if balance is None:
                return None
            free = safe_nonneg_float(balance.get("free"), default=None)
            locked = safe_nonneg_float(balance.get("locked"), default=None)
            if free is None or locked is None:
                return None
            return self.po.position_cost_basis(
                pair_norm or pair, free + locked, COST_BASIS_LOOKBACK_S, provider_name="binance")
        except Exception as e:  # noqa: BLE001
            self.log(f"  [TRAIL] cost_basis({pair}) unavailable ({e}) — warm-up from current price")
            return None

    # == TrailingCore ADAPTER contract ========================================
    def assets(self):
        for symbol in self.sym.symbols:
            spec = self._lookup_symbol_entry(TRAILING_INSTRUMENTS, symbol)
            if spec is not None:
                yield (symbol, spec.base, symbol, self.trail_pct_for(symbol))

    def begin_tick(self) -> bool:
        try:
            balances = self.api.get_account_assets_balances()
            # bapi currently represents an account-read failure as []; treating an
            # empty account identically is safe because there is nothing to protect.
            if not isinstance(balances, list) or not balances:
                self._balances = []
                self.log("  ! [TRAIL] empty or invalid balance snapshot — skipping the tick")
                return False
            self._balances = balances
            return True
        except Exception as e:  # noqa: BLE001
            self.log(f"  ! [TRAIL] balances unavailable ({e}) — skipping the tick")
            return False

    def free_qty(self, asset: str) -> float:
        bal = self._find_balance_row(self._balances, asset)
        return safe_nonneg_float(bal.get("free"), default=0.0) if bal is not None else 0.0

    def total_qty(self, asset: str) -> float:
        bal = self._find_balance_row(self._balances, asset)
        if bal is None:
            return 0.0
        f = safe_nonneg_float(bal.get("free", 0.0), default=None)
        l = safe_nonneg_float(bal.get("locked", 0.0), default=None)
        if f is None or l is None:
            return 0.0
        return safe_nonneg_float(f + l, default=0.0)

    def unlock_qty(self, pair: str) -> bool:
        """Cancel open SELL orders to release locked quantity for emergency stop."""
        try:
            cancel_fn = getattr(self.api, "cancel_open_orders", None)
            if callable(cancel_fn):
                spec = self._lookup_symbol_entry(TRAILING_INSTRUMENTS, pair)
                asset = spec.base if spec is not None else None
                if asset and self._balances:
                    bal = self._find_balance_row(self._balances, asset)
                    if bal is not None and safe_pos_float(bal.get("locked", 0.0), default=0.0) <= 0.0:
                        return False
                cancel_fn("SELL", pair)
                refreshed = self.api.get_account_assets_balances()
                if isinstance(refreshed, list) and refreshed:
                    self._balances = refreshed
                    return True
                return False
        except Exception as e:  # noqa: BLE001
            self.log(f"  ! [TRAIL] failed to cancel open orders for {pair}: {e}")
        return False

    def price(self, pair: str):
        return safe_pos_float(self.api.get_current_price(pair), None)

    def _submit_lifecycle_order(self, intent, persist, pair, side, qty, price,
                                kind="TRAIL", bypass_guards=None, market=False):
        """Submit through Instrument without losing its typed refusal state."""
        outcome_context = {}
        if bypass_guards is None:
            bypass_guards = {"all"} if side == "SELL" else {"math_reference"}

        def submit_once():
            response = self.po.place(
                pair, side, price, qty, force=True,
                kind=kind,
                bypass_guards=bypass_guards, smart=False,
                caller_owns_retry=True, wait_for_trend=(side != "SELL"),
                client_order_id=intent["client_order_id"],
                _outcome_context=outcome_context)
            return propagate_submission_refusal(response, outcome_context)

        return self._submit_and_clear_if_refused(intent, persist, submit_once)

    def run(self):
        mode = "⚠ ACTIVE (it really sells)" if self.enabled else "DRY RUN (it only logs)"
        self._run_trailing_loop(
            CHECK_SECONDS,
            [
                f"=== TRAILING STOP started — {mode} ===",
                "    coins/thresholds: "
                + ", ".join(f"{symbol}={trail}%" for symbol, _asset, _pair, trail in self.assets()),
            ],
        )


def main() -> int:
    args = parse_trailing_cli_args("Per-instrument trailing stop (Binance).", "binance_trailing")

    from binance_api import bapi as api
    ts = TrailingStop(api, po, sym)
    return ts.dispatch_cli(args)


if __name__ == "__main__":
    sys.exit(main())
