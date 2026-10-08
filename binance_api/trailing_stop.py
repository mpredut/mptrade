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

import argparse
import math
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # binance_api/ -> repository root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)   # also support direct execution (python binance_api/trailing_stop.py)

from instrument_registry import select_instruments
from providers.execution_audit import intent_client_order_id
from order_retry import (
    TrackedOrderLifecycle,
    propagate_submission_refusal,
)

from trailing_core import TrailingCore, should_sell  # noqa: E402  (re-export should_sell for tests/compatibility)
from botcore import (load_dotenv, required_bool_env, required_float_env,
                     required_int_env, single_instance)  # noqa: E402

DEFAULT_STATE = os.path.join(_ROOT, "cachedb", "trailing_state.json")


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
# Recent history is usable only when it explains the currently held inventory.
_COST_BASIS_LOOKBACK_S = 120 * 24 * 3600
TRAILING_ENABLED = required_bool_env("TRAILING_ENABLED")
SELL_FRACTION = required_float_env("TRAILING_SELL_FRACTION")
MIN_NOTIONAL_USD = 11.0
CHECK_SECONDS = required_float_env("TRAILING_CHECK_SECONDS")

# Re-buy after a crash stop-loss: the trailing component that sold with a bypass also
# buys back with a bypass, avoiding the profit guard whose long window would block
# re-entry. Trigger when price rebounds REBUY_BOUNCE_PCT% from the post-sale low,
# confirming the fall has stopped before entry. One tranche is supported currently;
# REBUY_TRANCHES is reserved for a future dip-DCA extension.
REBUY_ENABLED = required_bool_env("TRAILING_REBUY_ENABLED")
REBUY_BOUNCE_PCT = required_float_env("TRAILING_REBUY_BOUNCE_PCT")
REBUY_TRANCHES = required_int_env("TRAILING_REBUY_TRANCHES")
# Trend filters read cache_instant_trend through cacheManager. They act only on a
# CLEAR opposing signal; neutral/unknown does not block, reverting to behavior
# without a filter. Skip re-buy on a clear downtrend. Crash sells are unfiltered by
# default so the circuit breaker remains reliable; enable the sell filter to avoid
# selling while the instant trend is clearly up (anti-wick behavior).
REBUY_SKIP_IF_TREND_DOWN = required_bool_env("TRAILING_REBUY_SKIP_IF_TREND_DOWN")
SELL_SKIP_IF_TREND_UP = required_bool_env("TRAILING_SELL_SKIP_IF_TREND_UP")
# Minimum profit before trailing activates (0 means immediate activation as before).
# This prevents selling at a loss after a normal dip immediately following a purchase.
MIN_PROFIT_PCT = required_float_env("TRAILING_MIN_PROFIT_PCT")
# `trailing.rebuy = auto` follows a long-term trend: re-buy only while price is above
# its N-day SMA. N = TRAILING_REBUY_TREND_DAYS (calendar days of DAILY closes). The
# closed-candle average is cached per UTC day; the live comparison is never cached.
REBUY_TREND_DAYS = required_int_env("TRAILING_REBUY_TREND_DAYS")
if not 1 <= REBUY_TREND_DAYS < 1000:
    raise ValueError("TRAILING_REBUY_TREND_DAYS must be in [1, 999] for the kline request")


def _finite(value, *, name: str, minimum=None, maximum=None) -> float:
    """Normalize one financial/runtime input or fail before the live loop starts."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


class TrailingStop:
    """Binance adapter for TrailingCore, providing balance, price, sell/buy,
    trend APIs, and provider-specific logging. TrailingCore owns the state machine."""

    def __init__(self, api, po, sym, log=print, enabled=None,
                 sell_fraction=SELL_FRACTION, state_file=DEFAULT_STATE,
                 min_profit_pct=MIN_PROFIT_PCT):
        self.api = api
        self.po = po
        self.sym = sym
        self.log = log
        self.enabled = TRAILING_ENABLED if enabled is None else bool(enabled)
        self.sell_fraction = _finite(
            sell_fraction, name="TRAILING_SELL_FRACTION", minimum=0.0, maximum=1.0)
        min_profit_pct = _finite(
            min_profit_pct, name="TRAILING_MIN_PROFIT_PCT", minimum=0.0)
        _finite(CHECK_SECONDS, name="TRAILING_CHECK_SECONDS", minimum=0.1)
        _finite(REBUY_BOUNCE_PCT, name="TRAILING_REBUY_BOUNCE_PCT", minimum=0.0)
        if REBUY_TRANCHES != 1:
            raise ValueError("TRAILING_REBUY_TRANCHES currently supports only the value 1")
        self.state_file = state_file
        self._balances = []
        self._long_trend_cache = {}   # symbol -> (SMA, UTC day) for auto re-buy
        self.core = TrailingCore(
            self, log=log, enabled=self.enabled, state_file=state_file,
            min_notional=MIN_NOTIONAL_USD, rebuy_enabled=REBUY_ENABLED,
            rebuy_bounce_pct=REBUY_BOUNCE_PCT,
            rebuy_skip_if_trend_down=REBUY_SKIP_IF_TREND_DOWN,
            sell_skip_if_trend_up=SELL_SKIP_IF_TREND_UP,
            sell_fraction=sell_fraction, item_isolation=True,
            min_profit_pct=min_profit_pct)
        self.order_lifecycle = TrackedOrderLifecycle(
            self.po, provider_name="BINANCE", venue="Binance",
            missing_confirmations=1, retry_on_lookup_error=True,
            max_age_seconds=180,
        )

    # -- state delegated to the core; retained for --status and tests ----------
    def _load(self) -> dict:
        return self.core.load()

    def _save(self, state: dict):
        self.core.save(state)

    def trail_pct_for(self, symbol: str) -> float:
        return TRAIL_PCT[symbol]

    def rebuy_enabled_for(self, symbol: str) -> bool:
        """Per-coin re-buy switch (registry trailing.rebuy): on/off, or 'auto' to follow
        the long-term trend. A symbol not in the trailing registry uses the global
        default (TRAILING_REBUY_ENABLED)."""
        mode = REBUY_MODE_BY_SYMBOL.get(symbol)
        if mode is None:
            return REBUY_ENABLED
        if mode == "on":
            return True
        if mode == "off":
            return False
        return self._long_trend_up(symbol)                    # auto

    def rebuy_configured_for(self, symbol: str) -> bool:
        """Auto keeps a sold-position recovery intent even while its trend is down."""
        if REBUY_MODE_BY_SYMBOL.get(symbol) == "auto":
            return True
        return self.rebuy_enabled_for(symbol)

    def _long_trend_up(self, symbol: str) -> bool:
        """Compare a fresh price with a complete, contiguous closed-day SMA.

        Cache only the average, never its price-dependent verdict. Refresh after
        UTC midnight; missing/stale/malformed history or price blocks auto re-buy.
        """
        now = time.time()
        day = int(now // 86400)
        cached = self._long_trend_cache.get(symbol)
        try:
            if cached and cached[1] == day:
                sma = cached[0]
            else:
                klines = self.api.client.get_klines(
                    symbol=symbol, interval="1d", limit=REBUY_TREND_DAYS + 1) or []
                closed = [k for k in klines if float(k[6]) < day * 86400000]
                window = closed[-REBUY_TREND_DAYS:]
                if len(window) != REBUY_TREND_DAYS:
                    raise ValueError("insufficient completed daily history")
                closes = []
                for index, kline in enumerate(window):
                    expected_open = (day - REBUY_TREND_DAYS + index) * 86400000
                    close = float(kline[4])
                    if (float(kline[0]) != expected_open
                            or float(kline[6]) != expected_open + 86400000 - 1
                            or not math.isfinite(close) or close <= 0):
                        raise ValueError("stale, discontinuous or invalid daily history")
                    closes.append(close)
                sma = sum(closes) / len(closes)
                if not math.isfinite(sma):
                    raise ValueError("invalid daily average")
                self._long_trend_cache[symbol] = (sma, day)
            price = _finite(self.api.get_current_price(symbol), name="re-buy price", minimum=0)
            return price > 0 and price > sma
        except Exception as e:  # noqa: BLE001
            self.log(f"  [TRAIL] long-trend({symbol}) unavailable ({e}) — auto rebuy OFF")
            return False

    def cost_basis(self, pair: str):
        """Use reconciled BUY/SELL inventory for new-position warm-up, when known.

        Include locked quantity when matching the account to fills. On missing,
        stale, or inconsistent evidence, retain the core's first-observation fallback.
        Existing peaks, warm-up thresholds, and pending orders are never rewritten.
        """
        try:
            asset = TRAILING_INSTRUMENTS[pair].base
            balances = [row for row in self._balances if row.get("asset") == asset]
            if len(balances) != 1:
                return None
            balance = balances[0]
            free, locked = float(balance["free"]), float(balance["locked"])
            if not all(math.isfinite(v) and v >= 0 for v in (free, locked)):
                return None
            return self.po.position_cost_basis(
                pair, free + locked, _COST_BASIS_LOOKBACK_S, provider_name="binance")
        except Exception as e:  # noqa: BLE001
            self.log(f"  [TRAIL] cost_basis({pair}) unavailable ({e}) — warm-up from current price")
            return None

    def _free_qty(self, balances: list, asset: str) -> float:
        for bal in balances or []:
            if bal.get("asset") == asset:
                try:
                    qty = float(bal.get("free", 0.0))
                    return qty if math.isfinite(qty) and qty >= 0 else 0.0
                except (TypeError, ValueError, OverflowError):
                    return 0.0
        return 0.0

    # -- instant trend from cacheManager for optional filters ------------------
    def _trend_value(self, symbol: str) -> float:
        """Return instant-trend slope (>0 up, <0 down, 0 neutral/unknown).
        Errors return 0: optional filters do not block, but this is not a safety guarantee."""
        try:
            import cacheManager as cm
            snap = cm.get_short_trend_manager().get_snapshot(symbol)
            if snap:
                return float(snap.get('gradient_recent', snap.get('slope_small', 0.0)) or 0.0)
        except Exception:
            pass
        return 0.0

    # == TrailingCore ADAPTER contract ========================================
    def assets(self):
        for symbol in self.sym.symbols:
            if symbol in TRAILING_INSTRUMENTS:
                asset = TRAILING_INSTRUMENTS[symbol].base
                yield (symbol, asset, symbol, self.trail_pct_for(symbol))

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
            self.log(f"  ! [TRAIL] balante indisponibile ({e}) — skipping the tick")
            return False

    def free_qty(self, asset: str) -> float:
        return self._free_qty(self._balances, asset)

    def price(self, pair: str):
        value = self.api.get_current_price(pair)
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return value if math.isfinite(value) and value > 0 else None

    def trend(self, pair: str) -> float:
        return self._trend_value(pair)

    def _order_intent(self, action, key, asset, pair, qty, price, **metadata):
        intent_id = (
            f"trailing:binance:{key}:{action}:{qty:.8f}:{price:.8f}:"
            f"{metadata.get('anchor', price):.8f}"
        )
        return self.order_lifecycle.new_intent(
            intent_id=intent_id,
            client_order_id=intent_client_order_id("Binance", intent_id),
            symbol=pair, side="SELL" if action == "SELL" else "BUY",
            requested_qty=qty, requested_price=price, kind=f"TRAIL_{action}",
            metadata={"action": action, "asset": asset, **metadata},
        )

    def reconcile_pending(self, pending, persist):
        return self.order_lifecycle.reconcile(pending, persist=persist)

    def _submit_lifecycle_order(self, intent, persist, pair, side, price, qty, bypass_guards=None):
        """Submit through Instrument without losing its typed refusal state."""
        outcome_context = {}
        if bypass_guards is None:
            bypass_guards = {"all"} if side == "SELL" else {"math_reference"}

        def submit_once():
            is_all_bypassed = "all" in bypass_guards or "*" in bypass_guards
            response = self.po.place(
                pair, side, price, qty, force=True,
                bypass_guards=bypass_guards,
                bypass_profit_guard=is_all_bypassed, smart=False,
                caller_owns_retry=True, wait_for_trend=False,
                client_order_id=intent["client_order_id"],
                _outcome_context=outcome_context)
            return propagate_submission_refusal(response, outcome_context)

        result = self.order_lifecycle.submit(
            intent, persist=persist, submit=submit_once)
        if result.outcome == "refused":
            # No venue order exists; clear the lifecycle slot so the strategy can
            # make a fresh decision on a later tick instead of entering recovery.
            persist(None)
        return result

    def execute_sell(self, key, asset, pair, qty, price, peak, trail, persist):
        # July 30: use the single guarded proxy (self.po = market_api.api, .place()).
        # force=True sells at MARKET for reliable crash execution.
        # bypass_guards={"all"} bypasses profit/history/microstructure protection because this is a
        # STOP-LOSS below the last buy; otherwise the guard would block it. Daily limits
        # and cooldown remain active as before; the bypass skips only profit and weighting.
        intent = self._order_intent(
            "SELL", key, asset, pair, qty, price, anchor=peak, trail=trail)
        result = self._submit_lifecycle_order(
            intent, persist, pair, "SELL", price, qty, bypass_guards={"all"})
        if result.outcome == "refused":
            self.log(
                f"  🛑 [TRAIL] SELL REFUSED {pair} {qty} @ ~{price:.4f} "
                f"({result.intent.get('refusal_reason')})")
        else:
            self.log(
                f"  🛑 [TRAIL] SELL PENDING {pair} {qty} @ ~{price:.4f} "
                f"(peak {peak:.4f}, -{trail}%, "
                f"orderId={result.intent.get('order_id')})")
        return result

    def execute_rebuy(self, key, asset, pair, qty, price, rb, persist):
        # Re-buy runs with force=True for reliable fill on bounce, but guards against
        # falling knife cascades (liquidation cascades, taker dumps) by bypassing only
        # the historical math reference while keeping microstructure/AI guards active.
        intent = self._order_intent(
            "REBUY", key, asset, pair, qty, price,
            anchor=float(rb.get("low") or price),
            sell_price=float(rb.get("sell_price") or 0.0))
        result = self._submit_lifecycle_order(
            intent, persist, pair, "BUY", price, qty, bypass_guards={"math_reference"})
        if result.outcome == "refused":
            self.log(
                f"  🟢 [TRAIL] RE-BUY REFUSED {pair} {qty} @ ~{price:.4f} "
                f"({result.intent.get('refusal_reason')})")
        else:
            self.log(
                f"  🟢 [TRAIL] RE-BUY PENDING {pair} {qty} @ ~{price:.4f} "
                f"(orderId={result.intent.get('order_id')})")
        return result

    def log_dry_sell(self, key, asset, pair, qty, price, peak, trail) -> None:
        self.log(f"  🟡 [TRAIL][DRY] WOULD SELL {pair} {qty} @ ~{price:.4f} "
                 f"(varf {peak:.4f}, scadere >= {trail}%)  "
                 f"[set TRAILING_ENABLED=true for it to execute]")

    def log_dry_rebuy(self, key, asset, pair, qty, price, rb) -> None:
        self.log(f"  🟡 [TRAIL][DRY] WOULD RE-BUY {pair} {qty} @ ~{price:.4f}  "
                 f"(a pullback from the low {rb['low']:.4f})  [TRAILING_ENABLED=true for it to execute]")

    def log_hold(self, key, asset, pair, price, peak, stop_at, trail, free) -> None:
        self.log(f"  [TRAIL] {pair}: {price:.4f}  varf {peak:.4f}  "
                 f"sells below {stop_at:.4f} (-{trail}%)")

    def log_skip_rebuy_trend(self, asset) -> None:
        self.log(f"  [TRAIL] re-buy {asset} deferred — the instant trend is CLEARLY down (not catching the knife)")

    def log_skip_sell_trend(self, key, asset, pair, trail) -> None:
        self.log(f"  [TRAIL] {pair}: -{trail}% reached but the instant trend is UP — NOT selling (anti-wick)")

    def log_item_error(self, key, e) -> None:
        self.log(f"  ! [TRAIL] {key}: {e}")

    # -- one step / loop -------------------------------------------------------
    def check_once(self) -> None:
        self.core.check_once()

    def run(self):
        mode = "⚠ ACTIVE (it really sells)" if self.enabled else "DRY RUN (it only logs)"
        self.log(f"=== TRAILING STOP started — {mode} ===")
        self.log(f"    coins/thresholds: " +
                 ", ".join(f"{symbol}={trail}%" for symbol, _asset, _pair, trail in self.assets()))
        while True:
            try:
                self.check_once()
            except KeyboardInterrupt:
                return
            except Exception as e:  # noqa: BLE001
                self.log(f"  ! [TRAIL] eroare ciclu ({e}) — continui")
            time.sleep(CHECK_SECONDS)


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-instrument trailing stop (Binance).")
    ap.add_argument("--once", action="store_true", help="one check, then exit")
    ap.add_argument("--status", action="store_true", help="the current peaks and thresholds")
    args = ap.parse_args()
    if not args.status:
        single_instance("binance_trailing")

    from binance_api import bapi as api
    from providers.market_api import api as po   # single guarded .place() proxy; formerly bapi_placeorder
    import symbols as sym
    ts = TrailingStop(api, po, sym)

    if args.status:
        state = ts._load()
        for s, _asset, _pair, tr in ts.assets():
            st = state.get(s, {})
            peak = st.get("peak")
            mode = ("pending order" if st.get("pending_order") else
                    "warming up" if "warmup_at" in st else
                    "waiting for rebuy" if st.get("rebuy") else
                    "tracking" if peak else "uninitialized")
            stop = peak * (1 - tr / 100) if peak else None
            print(f"{s}: state={mode} peak={peak} trailing={tr}% threshold={stop} "
                  f"warmup_at={st.get('warmup_at')}")
        print("Persisted state only; this does not confirm a running bot or a fresh price.")
        print(f"ENABLED={ts.enabled} (a dry run if False)")
        return 0
    if args.once:
        ts.check_once()
        return 0
    ts.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
