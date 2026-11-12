import contextlib
import math
import os
import signal as _signal
import time
import datetime
import threading

# Local imports.
from notify_engine import alertnotifiers as alert  # noqa: F401 (re-export: test compat)
import cacheManager as cm
from intelligence.internal.state.volatility import get_kalman_state_path
import order_retry
from instrument_registry import (
    casefold_get,
    is_symbol_market_open,
    select_instruments,
    symbols_for,
    tracked_price_instruments,
)
import order_lifecycle
from cachepaths import clean_symbol_key, is_primary_stream_venue, split_symbol_assets
from providers.market_api import (
    api as mkt,  # Single guarded proxy (Instrument.place).
    resolve_runtime_market_regime_snapshot,
)
api = mkt
import shadow_signals
from state_io import append_text_line, read_json_dict_cached

from pricewindow import (  # noqa: F401 (re-export: test compat)
    PriceTrendAnalyzer,
    PriceWindow,
    RECENT_GRADIENT_SECONDS,
    WINDOW_SECONDS_BIG,
    WINDOW_SECONDS_SMALL,
    WindowAnalyzer,
)

# July 23: load tunable parameters from the versioned, secret-free
# config.env before reading any environment variables below.
# botcore.load_dotenv does not overwrite variables already set by the real
# environment (for example, a systemd EnvironmentFile); it only fills gaps.
from botcore import (PRICE_FRESHNESS_MAX_AGE_MS, PRICE_STREAM_POLL_INTERVAL_SEC,
                     _parse_bool_flag, bind_attr_call, exact_nonneg_int, exact_pos_int, extract_order_id,
                     format_price_log, load_dotenv, normalize_epoch_sec, normalize_side,
                     required_bool_env, required_float_env, required_int_env, resolve_now_ts,
                     safe_float, safe_nonneg_float, safe_pos_float, sanitize_pipe_field,
                     timeToHMS, touch_module_heartbeat)
load_dotenv()


TIME_SLEEP_GET_PRICE = PRICE_STREAM_POLL_INTERVAL_SEC  # Nominal price-collection sleep interval in seconds.
MIN_EVAL_INTERVAL_SEC = required_float_env("TRADEALL_MIN_EVAL_INTERVAL_SEC")
MAX_EVAL_INTERVAL_SEC = 30.0
ORDER_EXPIRY_HOURS = required_float_env("TRADEALL_ORDER_EXPIRY_HOURS")
EXP_TIME_BUY_ORDER = ORDER_EXPIRY_HOURS * 3600  # Configurable order expiration horizon.
EXP_TIME_SELL_ORDER = EXP_TIME_BUY_ORDER

# July 23: the parameters below were hard-coded constants. They now come from
# environment variables. Missing trading configuration aborts startup.
TREND_TO_BE_OLD_SECONDS = required_float_env("TRADEALL_TREND_OLD_HOURS") * 3600
# These values were calculated by u.calculate_difference_percent(60000, 60000-310)
# and (97000, 95000-377). They are stored as direct percentages with enough
# precision to reproduce those calls exactly (0.518...% / 2.481...%).
PRICE_CHANGE_THRESHOLD_EUR = required_float_env("TRADEALL_PRICE_CHANGE_THRESHOLD_PCT")
PRICE_CHANGE_THRESHOLD_BIG_EUR = required_float_env("TRADEALL_PRICE_CHANGE_THRESHOLD_BIG_PCT")

# TrendState.is_trend_a_minim_validated/is_trend_consistent_validated/is_trend_uniform_confirmed
# These thresholds specify how thoroughly a trend must be validated before it is
# considered reliable; see the TrendState methods below.
TREND_MIN_VALIDATED_SECONDS = required_float_env("TRADEALL_TREND_MIN_VALIDATED_SEC")
TREND_MIN_VALIDATED_CONFIRMS = required_int_env("TRADEALL_TREND_MIN_VALIDATED_CONFIRMS")
TREND_CONSISTENT_CONFIRMS = required_int_env("TRADEALL_TREND_CONSISTENT_CONFIRMS")
TREND_UNIFORM_RATE_THRESHOLD = required_float_env("TRADEALL_TREND_UNIFORM_RATE")
# logic(): the extreme-slope threshold used at four symmetric UP/DOWN sites. It
# bypasses normal validation and treats the trend as old for those branches.
SLOPE_EXTREME_THRESHOLD = required_float_env("TRADEALL_SLOPE_EXTREME_THRESHOLD")
SMALL_SLOPE_THRESHOLD = required_float_env("TRADEALL_SMALL_SLOPE_THRESHOLD")

# Proactive kinematic parameters (Velocity & Acceleration)
ACCEL_ENABLED = required_bool_env("TRADEALL_ACCEL_ENABLED")
ACCEL_WINDOW_SEC = required_float_env("TRADEALL_ACCEL_WINDOW_SEC")
ACCEL_MIN_VEL_SHORT = required_float_env("TRADEALL_ACCEL_MIN_VEL_SHORT")
ACCEL_MIN_ACC_SHORT = required_float_env("TRADEALL_ACCEL_MIN_ACC_SHORT")
ACCEL_MIN_VEL_MED = required_float_env("TRADEALL_ACCEL_MIN_VEL_MED")



# July 22: per-trend cooldown, based on real July 21-22 data and seven experiments
# in offline/research/tradeall_trigger_gate/. logic() previously fired on every
# evaluation while trend_state remained validated, even without a new event
# (TAO produced 186 BUY and zero SELL actions from one trend). Testing showed the
# cooldown reduces trades to a few per symbol and improves the result from
# -$57.56 (below buy-and-hold) to -$1.00 without changing start/confirmation rules.
# The updated behavior permits up to FIRE_MAX_PER_TREND confirmed executions per
# direction and trend, with FIRE_MIN_RETRY_INTERVAL_SEC between any two attempts,
# whether accepted or rejected. The interval is six minutes, reduced from 30.
FIRE_MIN_RETRY_INTERVAL_SEC = required_float_env("TRADEALL_FIRE_MIN_RETRY_MINUTES") * 60
FIRE_MAX_PER_TREND = required_int_env("TRADEALL_FIRE_MAX_PER_TREND")

# July 30: safeback_seconds for _fire_order. The same local expression previously
# appeared three times in logic(). This day-based window controls how far
# place_safe_order/if_place_safe_order searches own trades for the daily cap and
# profit-guard reference; see order_guard.py and bapi_placeorder.py.
FIRE_SAFEBACK_DAYS = required_float_env("TRADEALL_FIRE_SAFEBACK_DAYS")
FIRE_SAFEBACK_SEC = FIRE_SAFEBACK_DAYS * 24 * 3600 + 60

ROOT = os.path.dirname(os.path.abspath(__file__))
DECISIONS_LOG_DIR = os.path.join(ROOT, "logger")


def _validate_tradeall_config():
    """Fail startup before trading when timing/risk configuration is unsafe."""
    positive = {
        "TREND_TO_BE_OLD_SECONDS": TREND_TO_BE_OLD_SECONDS,
        "PRICE_CHANGE_THRESHOLD_EUR": PRICE_CHANGE_THRESHOLD_EUR,
        "PRICE_CHANGE_THRESHOLD_BIG_EUR": PRICE_CHANGE_THRESHOLD_BIG_EUR,
        "TREND_UNIFORM_RATE_THRESHOLD": TREND_UNIFORM_RATE_THRESHOLD,
        "SLOPE_EXTREME_THRESHOLD": SLOPE_EXTREME_THRESHOLD,
        "FIRE_MIN_RETRY_INTERVAL_SEC": FIRE_MIN_RETRY_INTERVAL_SEC,
        "FIRE_SAFEBACK_SEC": FIRE_SAFEBACK_SEC,
        "ORDER_EXPIRY_HOURS": ORDER_EXPIRY_HOURS,
        "MIN_EVAL_INTERVAL_SEC": MIN_EVAL_INTERVAL_SEC,
        "SMALL_SLOPE_THRESHOLD": SMALL_SLOPE_THRESHOLD,
        "ACCEL_WINDOW_SEC": ACCEL_WINDOW_SEC,
    }
    for name, value in positive.items():
        if safe_pos_float(value, default=None) is None:
            raise ValueError(f"tradeall config invalid: {name}={value!r}")
    if MIN_EVAL_INTERVAL_SEC > MAX_EVAL_INTERVAL_SEC:
        raise ValueError(
            f"tradeall config invalid: MIN_EVAL_INTERVAL_SEC={MIN_EVAL_INTERVAL_SEC!r} > MAX_EVAL_INTERVAL_SEC={MAX_EVAL_INTERVAL_SEC!r}"
        )
    nonnegative_counts = {
        "TREND_MIN_VALIDATED_CONFIRMS": TREND_MIN_VALIDATED_CONFIRMS,
        "TREND_CONSISTENT_CONFIRMS": TREND_CONSISTENT_CONFIRMS,
    }
    for name, value in nonnegative_counts.items():
        if exact_nonneg_int(value, default=None) is None:
            raise ValueError(f"tradeall config invalid: {name}={value!r}")
    if safe_nonneg_float(TREND_MIN_VALIDATED_SECONDS, default=None) is None:
        raise ValueError(
            f"tradeall config invalid: TREND_MIN_VALIDATED_SECONDS={TREND_MIN_VALIDATED_SECONDS!r}")
    if exact_pos_int(FIRE_MAX_PER_TREND, default=None) is None:
        raise ValueError(
            f"tradeall config invalid: FIRE_MAX_PER_TREND={FIRE_MAX_PER_TREND!r}")


_validate_tradeall_config()


def log_decision(symbol, event, **fields):
    """Write one condensed, pipe-delimited row per real trend transition.

    Only ``trend_start`` is recorded. The file rotates daily like other logger/
    output and is observational, so it does not affect trading logic.
    """
    try:
        os.makedirs(DECISIONS_LOG_DIR, exist_ok=True)
        path = os.path.join(DECISIONS_LOG_DIR,
                             f"tradeall_decisions_{datetime.date.today().isoformat()}.log")
        cols = [time.time(), symbol, event,
                fields.get("state", ""), fields.get("old_state", ""), fields.get("price", ""),
                fields.get("prev_confirm_count", "")]
        append_text_line(path, "|".join(sanitize_pipe_field(c) for c in cols))
    except Exception as e:
        print(f"[log_decision] error writing decisions log: {e}")


# Kalman gate (approved July 19): determine whether model orders reach real funds.
# The registry mode controls each order. ``strict`` allows BUY only on Kalman UP
# and SELL only on DOWN; ``permissive`` blocks only countertrend orders; ``off``
# restores the previous behavior. Fail open when the Kalman signal is missing or
# older than five minutes so a signal failure cannot stop trading.
_kalman_ref = None                      # Set by TrendCoordinator.__init__.
GATE_OUTCOME_LOG = None                 # Backtests redirect this away from the live log.
GATE_STALE_SEC = 300


# All per-instrument execution choices come from instruments.conf across all
# enabled venues (Binance, Kraken, Hyperliquid, Trading212). An enabled
# instrument is trend-tracked even when it opts out of TradeAll order attempts.
# No opted-in symbols means no TradeAll orders; it is not a missing configuration.
PROVIDER_BY_SYMBOL = {
    spec.symbol.strip().upper(): spec.provider.strip()
    for spec in select_instruments(only_enabled=True).values()
}
KALMAN_MODES = {
    spec.symbol: spec.setting("tradeall.kalman_mode")
    for spec in select_instruments(role="tradeall_fire").values()
}
TRADEALL_FIRE_SYMBOLS = set(KALMAN_MODES)
KALMAN_PRIMARY_SYMBOLS = set(symbols_for(role="kalman_primary"))
KALMAN_BYPASS_SELL_REASONS = frozenset({"accel_plunge_down", "small_window_plunge_down"})


def _provider_kw(symbol) -> dict:
    prov = casefold_get(PROVIDER_BY_SYMBOL, str(symbol or "").strip().upper())
    if prov and not is_primary_stream_venue(prov):
        return {"provider_name": prov}
    return {}


def _resolve_tradeall_symbol(symbol, allowed_set=TRADEALL_FIRE_SYMBOLS, *, preserve_raw=False):
    sym_u = str(symbol or "").strip().upper()
    sym_clean = clean_symbol_key(sym_u)
    if sym_u in allowed_set:
        return sym_u
    if sym_clean in allowed_set:
        return sym_clean
    for s in allowed_set:
        if isinstance(s, str) and s.upper() in (sym_u, sym_clean):
            return s
    return symbol if preserve_raw else sym_u


def _kalman_gate_blocks(symbol, action, reason=None):
    # An allowed symbol must have an explicit mode; there is no default policy.
    action = normalize_side(action) or str(action or "").upper()
    symbol = _resolve_tradeall_symbol(symbol, KALMAN_MODES, preserve_raw=True)
    mode = KALMAN_MODES.get(symbol) or casefold_get(KALMAN_MODES, symbol, "off") or "off"
    if mode == "off" or _kalman_ref is None:
        return False, mode, None        # Gate disabled -> never blocks (not a data state).
    if action == "SELL" and reason in KALMAN_BYPASS_SELL_REASONS:
        return False, mode, None        # Fast kinematic plunge exits bypass Kalman lag.
    try:
        trend, age = _kalman_ref.current_trend(symbol)
        age_f = safe_nonneg_float(age, default=None)
    except Exception:
        # No confirming signal (read failed): fail CLOSED for a BUY -- do not add
        # exposure blind -- but never for a SELL, so a risk-reducing exit is never
        # trapped by missing data.
        return action == "BUY", mode, None
    if trend is None or age_f is None or age_f > GATE_STALE_SEC:
        return action == "BUY", mode, None   # Absent/stale signal: block a blind BUY, allow a SELL.
    wanted = 1 if action == "BUY" else -1
    if mode == "permissive":
        return trend == -wanted, mode, trend
    return trend != wanted, mode, trend  # strict


def _fire_order(symbol, action, price, reason, **kwargs):
    """Submit an order through the single guarded proxy.

    Since July 30 this uses ``mkt.place`` instead of ``po.place_order_smart``;
    Instrument.place centrally records provider acceptance or refusal in
    order_outcomes; acceptance is not a confirmed fill.
    The Kalman gate must pass before execution. No explicit quantity is supplied:
    ``qty=None`` uses the QuantityDecision maximum under Binance policy, clamped
    to the real balance.
    """
    action, symbol = normalize_side(action) or str(action or "").upper(), _resolve_tradeall_symbol(symbol)
    price_f = safe_pos_float(price, default=None)
    if price_f is None:
        print(f"[TRADEALL] order refused: price invalid {price!r}")
        return None
    price = price_f
    if action not in {"BUY", "SELL"}:
        print(f"[TRADEALL] order refused: side invalid {action!r}")
        return None

    # Symbol allowlist: a coin can be trend-tracked without being traded here
    # (for example a manual position guarded by the trailing stop). This is the
    # single choke point for both the logic() and Kalman-primary firing paths.
    if symbol not in TRADEALL_FIRE_SYMBOLS:
        print(f"[TRADEALL] {action} {symbol} skipped: not in TRADEALL_FIRE_SYMBOLS "
              f"(trend-tracked only; exit managed elsewhere)")
        return None

    blocked, mode, trend = _kalman_gate_blocks(symbol, action, reason=reason)
    if blocked:
        print(f"[KALMAN-GATE] {action} {symbol} BLOCKED (kalman_trend={trend}, mode={mode}, reason={reason})")
        try:
            logger_fn = GATE_OUTCOME_LOG or order_lifecycle.log_order_outcome
            logger_fn(symbol, action, price, None,
                      "refused", f"kalman_gate_{mode}(trend={trend})", reason)
        except Exception as _e:  # noqa: BLE001
            print(f"[KALMAN-GATE] journal error ({_e}) — block remains")
        return None
    return mkt.place(
        symbol,
        action,
        price,
        None,
        motivation=reason,
        **{**_provider_kw(symbol), **kwargs},
    )


_symbol_order_ids = {}
_SYMBOL_TICK_SIZE_CACHE: dict[str, float] = {}


def track_and_place_order(action, symbol, count, proposed_price, current_price, order_ids=None,
                          reason="trend_signal", **kwargs):
    """Execute trading strategy with order tracking, expiration, and replacement.

    Restores the trading strategy logic of the original track_and_place_order()
    while routing placements through the guarded proxy (_fire_order / mkt.place).
    Cancels prior active orders from previous signals (alerting if already executed),
    cancels expired orders on the venue, applies side-specific strategic pricing,
    and records newly placed order IDs.
    """
    action, symbol = normalize_side(action) or str(action or "").upper(), _resolve_tradeall_symbol(symbol)
    print(f"Iteration {count} generated price {proposed_price} versus {current_price}")

    if order_ids is None:
        if symbol not in _symbol_order_ids:
            _symbol_order_ids[symbol] = []
        order_ids = _symbol_order_ids[symbol]

    if action == 'HOLD':
        return order_ids

    quote = (split_symbol_assets(symbol) or ("", ""))[1]

    prop_f = safe_pos_float(proposed_price, default=None)
    curr_f = safe_pos_float(current_price, default=None)
    eff_min_proc, _, _, _ = order_lifecycle.resolve_dynamic_reprice_bounds(
        symbol,
        curr_f if curr_f is not None else 0.0,
        market_api=mkt,
        tick_cache=_SYMBOL_TICK_SIZE_CACHE,
        skip_live_tick_lookup=hasattr(_fire_order, "_mock_name"),
    )
    num_orders, price_step = 1, eff_min_proc
    if prop_f is None or curr_f is None:
        base_price = math.nan
    else:
        entry_offset = eff_min_proc / 100.0
        base_price = (
            min(prop_f, curr_f * (1.0 - entry_offset))
            if action == "BUY"
            else max(prop_f, curr_f * (1.0 + entry_offset))
        )

    # Pre-submit gate check: if _fire_order will deterministically refuse this
    # signal (invalid parameters, not in TRADEALL_FIRE_SYMBOLS, or blocked by
    # the Kalman gate), delegate directly to _fire_order for logging/journaling
    # and return [] without canceling existing live orders on the venue.
    if not hasattr(_fire_order, "_mock_name"):
        if (
            action not in {"BUY", "SELL"}
            or safe_pos_float(base_price, default=None) is None
            or symbol not in TRADEALL_FIRE_SYMBOLS
            or _kalman_gate_blocks(symbol, action, reason=reason)[0]
        ):
            _fire_order(symbol, action, base_price, reason, **kwargs)
            return []

    # Cancel any existing active orders from previous iterations (including outbox-accepted orders)
    outbox_order_ids = []
    try:
        if (
            not hasattr(_fire_order, "_mock_name")
            or hasattr(order_retry.get_accepted_order_ids_for_owner, "_mock_name")
        ):
            outbox_order_ids = order_retry.get_accepted_order_ids_for_owner(symbol, "tradeall")
    except Exception:
        outbox_order_ids = []

    to_cancel = []
    existing_str = set()
    for raw_oid in list(order_ids) + list(outbox_order_ids):
        clean_oid = extract_order_id(raw_oid, default=None)
        if clean_oid and clean_oid not in existing_str:
            to_cancel.append(clean_oid)
            existing_str.add(clean_oid)

    if to_cancel or order_ids:
        failed_cancels = []
        for order_id in to_cancel:
            try:
                if mkt.cancel_order(symbol, str(order_id), **_provider_kw(symbol)) is False:
                    failed_cancels.append(order_id)
                    continue
                try:
                    order_retry.resolve_accepted_order(symbol, order_id)
                except Exception:
                    pass
            except Exception as _e:
                print(f"[TRADEALL] cancel existing order {order_id} (or filled): {_e}")
        order_ids.clear()
        order_ids.extend(failed_cancels)


    # Cancel expired orders on venue
    exp_time = EXP_TIME_BUY_ORDER if action == 'BUY' else EXP_TIME_SELL_ORDER
    try:
        api.cancel_expired_orders(
            action,
            symbol,
            exp_time,
            allowed_owners={"tradeall"},
            **_provider_kw(symbol),
        )
    except Exception as _e:
        print(f"[TRADEALL] cancel_expired_orders error for {symbol}: {_e}")

    print(f"{action} price: {format_price_log(base_price)} {quote}".strip())

    direction = -1 if action == "BUY" else 1
    for i in range(num_orders):
        adjusted_price = base_price * (1 + direction * i * price_step / 100)
        order = _fire_order(symbol, action, adjusted_price, reason, **kwargs)
        if order:
            oid = extract_order_id(order, default=None)
            if oid:
                raw_id = order.get("orderId") or order.get("id") if isinstance(order, dict) else None
                if not isinstance(raw_id, bool) and isinstance(raw_id, int) and raw_id != 0:
                    order_ids.append(raw_id)
                else:
                    order_ids.append(oid)

    return order_ids


class TrendState:
    def __init__(
        self,
        expiration_trend_time,
        fresh_trend_time,
        now_fn=time.time,
        *,
        min_confirm_interval_sec: float = 0.0,
    ):
        self.state = 'HOLD'
        self.old_state = self.state
        self.expired = False
        self.start_time = None
        self.end_time = None
        self.last_confirmation_time = None
        self.confirm_count = 0
        self.expiration_trend_time = expiration_trend_time  # Maximum seconds between confirmations.
        self.fresh_trend_time = fresh_trend_time            # Freshness threshold.
        self._now = now_fn   # Real clock by default; replay injects each tick's time (A5).
        self._min_confirm_interval = safe_nonneg_float(min_confirm_interval_sec, default=0.0) or 0.0
        self._last_counted_confirm_ts = None
        # Per-trend cooldown: _confirmed_count_{up,down} is a legacy field name
        # counting accepted submissions, not fills. Stop at
        # FIRE_MAX_PER_TREND until a new trend. _last_attempt_* records the last
        # accepted or rejected attempt, and every retry must wait at least
        # FIRE_MIN_RETRY_INTERVAL_SEC.
        (self._confirmed_count_up, self._confirmed_count_down, self._last_attempt_up_ts, self._last_attempt_down_ts) = (0, 0, None, None)

    @staticmethod
    def _norm_ts(ts):
        return normalize_epoch_sec(ts, default=None, allow_zero=True)

    def _now_sec(self):
        return resolve_now_ts(self._now(), allow_zero=True)

    def start_trend(self, new_state):
        assert new_state in ['UP', 'DOWN', 'HOLD'], "Invalid trend state"
        self.old_state = self.state
        self.state = new_state
        self.start_time = self._now_sec()
        self.last_confirmation_time = self.start_time
        self._last_counted_confirm_ts = self.start_time
        self.confirm_count = 1
        self.end_time = None
        self.expired = False
        (self._confirmed_count_up, self._confirmed_count_down, self._last_attempt_up_ts, self._last_attempt_down_ts) = (0, 0, None, None)
        print(f"Start of {self.state} trend at {timeToHMS(self.start_time)}")
        return self.old_state

    def fire_limit_reached(self, direction):
        count = self._confirmed_count_up if direction == 'UP' else self._confirmed_count_down
        return count >= FIRE_MAX_PER_TREND

    def mark_confirmed(self, direction):
        attr = "_confirmed_count_up" if direction == "UP" else "_confirmed_count_down"
        setattr(self, attr, getattr(self, attr) + 1)

    def can_retry_fire(self, direction):
        last = self._norm_ts(self._last_attempt_up_ts if direction == 'UP' else self._last_attempt_down_ts)
        if last is None:
            return True
        elapsed = self._now_sec() - last
        return elapsed < 0 or elapsed >= FIRE_MIN_RETRY_INTERVAL_SEC

    def mark_fire_attempt(self, direction):
        setattr(self, "_last_attempt_up_ts" if direction == "UP" else "_last_attempt_down_ts", self._now_sec())

    def confirm_trend(self):
        assert self.start_time is not None, "Trend must be started before confirming"
        now = self._now_sec()
        self.last_confirmation_time = now
        last_counted = self._norm_ts(self._last_counted_confirm_ts)
        if (
            self._min_confirm_interval <= 0.0
            or last_counted is None
            or (now - last_counted) < 0.0
            or (now - last_counted) >= self._min_confirm_interval
        ):
            self.confirm_count += 1
            self._last_counted_confirm_ts = now
            print(f"{self.confirm_count} times trend confirmed: {self.state} at {timeToHMS(self.last_confirmation_time)}")
        return self.confirm_count

    def get_started_trend_time(self):
        st = self._norm_ts(self.start_time)
        if st is None:
            return 0
        return self._now_sec() - st


    def is_trend_fresh(self, fresh_trend_time=None):
        if fresh_trend_time is None:
            fresh_trend_time = self.fresh_trend_time
        st = self._norm_ts(self.start_time)
        assert st is not None, "Trend must be started before checking freshness"

        elapsed_time = self._now_sec() - st
        if elapsed_time < fresh_trend_time:
            return True

        print(f"No fresh trend! Current trend start {math.trunc(elapsed_time)} sec back > {math.trunc(fresh_trend_time)} sec ")
        return False



    def is_trend_a_minim_validated(self) :
        st = self._norm_ts(self.start_time)
        lct = self._norm_ts(self.last_confirmation_time)
        if st is None or lct is None:
            return False
        return (lct - st > TREND_MIN_VALIDATED_SECONDS
                and self.confirm_count > TREND_MIN_VALIDATED_CONFIRMS)

    def is_trend_consistent_validated(self) :
         # Originally 14 confirmations per minute * 3; effectively six per minute.
        return self.confirm_count > TREND_CONSISTENT_CONFIRMS and self.is_trend_uniform_confirmed()

    def is_trend_uniform_confirmed(self):
        if not self.is_trend_a_minim_validated() :
            return False

        trend_duration = self.get_started_trend_time() # self.get_confirmed_trend_duration()
        if trend_duration <= 0:
            return False
        rate = self.confirm_count * 2.5 * TIME_SLEEP_GET_PRICE / trend_duration
        print(f"uniform rate is {rate} <> {TREND_UNIFORM_RATE_THRESHOLD}")
        # Ten confirmations per 1.5 minutes.
        return rate > TREND_UNIFORM_RATE_THRESHOLD

    def is_started_trend_older_than(self, old_trend_time):
        return self.get_started_trend_time() > old_trend_time

    def check_trend_expiration(self):
        if self.expired:
            return True
        lct = self._norm_ts(self.last_confirmation_time)
        if lct:
            time_since_last_confirmation = self._now_sec() - lct
            if time_since_last_confirmation > self.expiration_trend_time:
                print(f"Trend expired: {self.state}. Time since last confirmation: {time_since_last_confirmation} seconds")
                self.old_state = self.state
                self.end_time = self.last_confirmation_time
                print(f"Trend ended: {self.state} at {timeToHMS(self.end_time)} after {self.confirm_count} confirmations.")
                self.old_confirm_count = self.confirm_count
                self.state = 'HOLD'
                self.confirm_count = 0
                self.expired = True
                return self.expired
        return False #self.expired

    def _confirm_count_for(self, target_state):
        if self.check_trend_expiration():
            return self.confirm_count if target_state == "HOLD" else 0
        return self.confirm_count if self.state == target_state else 0

    is_trend_up = bind_attr_call("_confirm_count_for", "UP")
    is_trend_down = bind_attr_call("_confirm_count_for", "DOWN")
  


# ``sell_recommendation.csv`` was removed. Trend signals are published as per-symbol
# snapshots by ``CacheShortTrendManager`` for cross-process consumers.



def _fire_trend_order(enable, symbol, trend_state, direction, action, price, reason, regime_ctx=None):
    """Enforce the per-trend cooldown and accepted-submission limit.

    A non-None result means the provider accepted the submission; it is not a
    confirmed fill. Gate, weight-limit, or budget rejection imposes the retry
    interval but does not block the trend permanently. Permit up to
    FIRE_MAX_PER_TREND accepted submissions per direction and trend.
    """
    if not enable:
        return None
    if trend_state.fire_limit_reached(direction):
        return None
    if not trend_state.can_retry_fire(direction):
        return None
    trend_state.mark_fire_attempt(direction)
    result = track_and_place_order(
        action, symbol, trend_state.confirm_count, price, price,
        reason=reason, safeback_seconds=FIRE_SAFEBACK_SEC, force=False,
        cancelorders=True, hours=0.3, regime_context=regime_ctx)
    if result:
        # The provider accepted a submission. This is intentionally a throttle
        # count, not a claim that the order filled.
        trend_state.mark_confirmed(direction)
    return result


def _dir_action(is_up: bool) -> tuple[str, str]:
    return ("UP", "BUY") if is_up else ("DOWN", "SELL")


def logic_small(win, enable, symbol, gradient, slope, trend_state, current_price, regime_ctx=None, price_window=None):
    """Evaluate fast-momentum and short-term acceleration impulses on the small price window."""
    trend_state.check_trend_expiration()

    def _advance_and_fire_small_window(is_up: bool, reason: str, log_prefix: str, log_detail: str):
        direction, action = _dir_action(is_up)
        print(f"[SMALL-WINDOW] {log_prefix} {action} {'surge' if is_up else 'plunge'}: {log_detail}")
        if trend_state.is_trend_up() if direction == "UP" else trend_state.is_trend_down():
            trend_state.confirm_trend()
        else:
            trend_state.start_trend(direction)
        return _fire_trend_order(
            enable, symbol, trend_state, direction, action, current_price,
            reason, regime_ctx=regime_ctx,
        )

    if (gradient > 0 and slope > SMALL_SLOPE_THRESHOLD) or (gradient < 0 and slope < -SMALL_SLOPE_THRESHOLD):
        is_up = gradient > 0
        cmp_str = f"> {SMALL_SLOPE_THRESHOLD}" if is_up else f"< -{SMALL_SLOPE_THRESHOLD}"
        reason = "small_window_surge_up" if is_up else "small_window_plunge_down"
        return _advance_and_fire_small_window(
            is_up, reason, "Momentum", f"gradient={gradient}, slope={slope} {cmp_str}",
        )

    if ACCEL_ENABLED and price_window is not None and (
        (gradient > 0 and slope >= PRICE_CHANGE_THRESHOLD_EUR)
        or (gradient < 0 and slope <= -PRICE_CHANGE_THRESHOLD_EUR)
    ):
        is_up = gradient > 0
        sign = 1.0 if is_up else -1.0
        try:
            vel_short, acc_short = price_window.calculate_acceleration(window_sec=ACCEL_WINDOW_SEC)
            vel_med, _ = price_window.calculate_acceleration(window_sec=max(ACCEL_WINDOW_SEC * 2.0, 180.0))
            if (
                sign * vel_short >= ACCEL_MIN_VEL_SHORT
                and sign * acc_short >= ACCEL_MIN_ACC_SHORT
                and sign * vel_med >= ACCEL_MIN_VEL_MED
            ):
                reason = "accel_surge_up" if is_up else "accel_plunge_down"
                return _advance_and_fire_small_window(
                    is_up,
                    reason,
                    "Acceleration",
                    (
                        f"gradient={gradient}, slope={slope:.3f}, "
                        f"vel_90s={vel_short:.3f}%/m, acc_90s={acc_short:.3f}%/m^2, vel_180s={vel_med:.3f}%/m"
                    ),
                )
        except Exception as e:
            print(f"[SMALL-WINDOW] Acceleration calculation error for {symbol}: {e}")
    return None


def logic(win, enable, symbol, gradient, slope, trend_state, current_price, regime_ctx=None):
    def _fire_once(direction, action, reason):
        return _fire_trend_order(enable, symbol, trend_state, direction, action,
                                 current_price, reason, regime_ctx=regime_ctx)

    print(f"LOGIC gradient={gradient}, slope={slope}")

    # Check for trend confirmation or initiation
    if (gradient > 0 and slope < 0) or (gradient < 0 and slope > 0):
        direction, action = _dir_action(gradient > 0)
        label = "DOWN" if direction == "UP" else "UP"
        print(f"LARGE DIFFERENCE {win} {label}!")
        if trend_state.is_trend_up() if direction == "UP" else trend_state.is_trend_down():
            trend_state.confirm_trend()
            if trend_state.is_trend_uniform_confirmed() and trend_state.is_trend_fresh():
                _fire_once(direction, action, f"trend_confirmed_{direction.lower()}")
                print(f"place_order_smart {action}")
        else:
            prev_confirm_count = trend_state.confirm_count
            old_trend = trend_state.start_trend(direction)
            log_decision(symbol, "trend_start", state=direction, old_state=old_trend, price=current_price,
                         prev_confirm_count=prev_confirm_count)

    # Originally 18 confirmations per minute * 3; effectively six per minute.
    if (slope <= 0 and trend_state.is_trend_up()) or (slope >= 0 and trend_state.is_trend_down()):
        direction, action = _dir_action(bool(trend_state.is_trend_up()))
        if (trend_state.is_trend_consistent_validated()
        or trend_state.is_started_trend_older_than(TREND_TO_BE_OLD_SECONDS)):
            print(f"ATTENTION {action} ALL {win} .... ")
            _fire_once(direction, action, f"consistent_or_old_{direction.lower()}")

    # Extreme slope counter-trend exhaustion / reversal case
    if (slope <= -SLOPE_EXTREME_THRESHOLD and trend_state.is_trend_down()) or (slope >= SLOPE_EXTREME_THRESHOLD and trend_state.is_trend_up()):
        direction, action = _dir_action(slope <= -SLOPE_EXTREME_THRESHOLD)
        if (trend_state.is_trend_consistent_validated()
        and trend_state.is_started_trend_older_than(TREND_TO_BE_OLD_SECONDS)):
            print(f"ATTENTION 3: {action} ALL {win} .... ")
            reason = "slope<=-5.1_and_old_down" if direction == "UP" else "slope>=5.1_and_old_up"
            _fire_once(direction, action, reason)


# Function to handle the price logic for a specific currency.
# Windows update autonomously through Cache24 subscriptions; this path only evaluates.
# Return the trend snapshot that ``TrendCoordinator`` will cache.
def handle_symbol(symbol, current_price, price_window, price_window_big,
                  analyzer, analyzer_big, trend_state, trend_state_big,
                  regime_ctx=None):

    # Refresh the observed sampling rate from ``CacheCurrentPriceManager``.
    try:
        actual_rate = cm.get_current_price_manager().get_sample_rate(
            symbol, fallback=TIME_SLEEP_GET_PRICE)
        price_window.set_sample_rate(actual_rate)
        price_window_big.set_sample_rate(actual_rate)
    except Exception:
        pass

    slope, pos = analyzer.check_price_change(PRICE_CHANGE_THRESHOLD_EUR)
    print(f"small slope {slope}")
    gradient, gradient_coff, slope_full, gradient_recent = price_window.get_short_trend()

    if slope * gradient < 0:
        print(f"ALERT slope1 = {slope} gradient = {gradient}")

    if regime_ctx is None:
        try:
            regime_ctx = mkt.market_regime_context(symbol, **_provider_kw(symbol))
        except Exception as e:
            print(f"[TRADEALL] Error resolving regime context for {symbol}: {e}")
    if regime_ctx is not None:
        print(f"[TRADEALL] {symbol} resolved trend: {getattr(regime_ctx, 'resolved_trend', None)}")

    # SMALL ONE!!
    small_placed = logic_small("SMALL", True, symbol, gradient, slope, trend_state, current_price,
                               regime_ctx=regime_ctx, price_window=price_window)

    # BIG ONE!!!
    slope_big, price_diff = analyzer_big.check_price_change(PRICE_CHANGE_THRESHOLD_BIG_EUR)
    logic("BIG", not bool(small_placed), symbol, gradient, slope_big, trend_state_big, current_price, regime_ctx=regime_ctx)

    # Cross-process snapshot aligned with CacheShortTrendManager SSOT.
    eps_fn = getattr(price_window, "get_noise_epsilon", None)
    eps_val = eps_fn() if callable(eps_fn) else 0.0
    n_samp = len(getattr(price_window, "prices", ()))
    win_sec = safe_pos_float(getattr(price_window, "window_seconds", None), default=222.0)
    sr_sec = safe_pos_float(getattr(price_window, "sample_rate_sec", None), default=1.0)
    return {
        "symbol": symbol,
        "final_trend": gradient,
        "growth_coefficient": gradient_coff,
        "slope_full": slope_full,
        "gradient_recent": gradient_recent,
        "slope_small": slope,
        "slope_big": slope_big,
        "pos": pos,
        "epsilon": eps_val,
        "n_samples": n_samp,
        "window_seconds": win_sec,
        "sample_rate_sec": sr_sec,
        "current_price": current_price,
        "ts": time.time(),
    }


# ============================================================================
# TrendCoordinator -- event-driven + heartbeat.
#
# The WS-to-Cache24 source updates windows autonomously. Each tick marks a symbol dirty;
# evaluation occurs only after ``MIN_EVAL_INTERVAL_SEC`` has elapsed.
#       - or MAX_EVAL_INTERVAL_SEC has elapsed (heartbeat prevents sparse evaluation)
# Cached lookup is O(1), and the single-threaded loop prevents reentrant placement.
# ============================================================================


class TrendCoordinator:
    """Coordinate event-driven trading evaluation with a heartbeat.

    ``CacheShortTrendManager`` owns the windows, trend-state calculations,
    and cross-process cache. This coordinator consumes them and makes
    ``handle_symbol``/``logic`` decisions.
    """
    def __init__(self, symbols, instant_mgr, current_price_mgr, cache24_managers=None,
                 min_interval=MIN_EVAL_INTERVAL_SEC, max_interval=MAX_EVAL_INTERVAL_SEC):
        min_i = safe_pos_float(min_interval, default=None)
        max_i = safe_pos_float(max_interval, default=None)
        if min_i is None or max_i is None or min_i > max_i:
            raise ValueError("TrendCoordinator intervals must satisfy 0 < min <= max")
        self.symbols = list(dict.fromkeys(symbols))
        self._canonical_symbols = {str(s).strip().upper(): s for s in self.symbols}
        self.instant_mgr = instant_mgr
        self.current_price_mgr = current_price_mgr
        self.min_interval = min_i
        self.max_interval = max_i

        self._event = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._dirty = {s: True for s in self.symbols}      # Force an initial evaluation.
        self._last_eval = {s: 0.0 for s in self.symbols}

        # shadow_signals originally combined a Kalman trend and adaptive volatility.
        # Only adaptive values remain observational because tradeall has no
        # reentry/DCA consumer. The Kalman trend moved to real trading on July 19:
        # KALMAN_MODES gates every order and primary Kalman starts orders for
        # KALMAN_PRIMARY_SYMBOLS. Guard setup and calls so a failure cannot stop trading.
        try:
            self._kalman_state = shadow_signals.KalmanStateTracker(
                state_path=get_kalman_state_path(),
                read_only=True,
            )
            global _kalman_ref
            _kalman_ref = self._kalman_state   # _fire_order's gate reads this signal.
            print(f"[KALMAN-GATE] active, modes={KALMAN_MODES}")
        except Exception as _e:  # noqa: BLE001
            print(f"[TrendCoordinator] shadow_signals unavailable (continuing without it): {_e}")
            self._kalman_state = None

        self._regime_ctx_memo = {}
        self.trend_states = {}
        self.trend_states_big = {}
        for symbol in self.symbols:
            self.trend_states[symbol] = self._new_trend_state()
            self.trend_states_big[symbol] = self._new_trend_state()
            # Windows and the fast channel live in ``instant_mgr``; subscribe here.
            # Use Cache24 only for its evaluation signal (dirty flag and event).
            if cache24_managers is not None:
                c24 = (
                    casefold_get(cache24_managers, symbol)
                    if isinstance(cache24_managers, dict)
                    else cache24_managers[symbol]
                )
                if c24 is not None:
                    c24.subscribe_price(self)

    def _new_trend_state(self) -> TrendState:
        eff_min_interval = safe_pos_float(getattr(self, "min_interval", None), default=TIME_SLEEP_GET_PRICE)
        min_confirm = TIME_SLEEP_GET_PRICE if eff_min_interval < TIME_SLEEP_GET_PRICE else 0.0
        return TrendState(
            expiration_trend_time=2.7 * 60,
            fresh_trend_time=3.7 * 60,
            min_confirm_interval_sec=min_confirm,
        )

    def _resolve_symbol(self, symbol: str) -> str:
        if isinstance(symbol, str):
            canon = self._canonical_symbols.get(symbol.strip().upper())
            if canon is not None:
                return canon
        return _resolve_tradeall_symbol(symbol)

    # Cache24 subscriber signal wakes evaluation.
    def on_price_update(self, symbol: str, ts_ms: int, price: float) -> None:
        sym = self._resolve_symbol(symbol)
        with self._lock:
            if sym not in self._dirty:
                return
            self._dirty[sym] = True
        self._event.set()

    # -- Decide whether the symbol is due for evaluation. ----------------------
    def _is_due(self, symbol, now):
        symbol = self._resolve_symbol(symbol)
        elapsed = now - self._last_eval.get(symbol, 0.0)
        if elapsed < 0:
            self._last_eval[symbol] = 0.0
            elapsed = now
        if elapsed >= self.max_interval:          # heartbeat
            return True
        with self._lock:
            dirty = self._dirty.get(symbol, False)
        return dirty and elapsed >= self.min_interval   # floor

    def _next_wait_timeout(self, now: float) -> float:
        wait_sec = self.max_interval
        with self._lock:
            dirty_state = [(s, self._dirty.get(s, False)) for s in self.symbols]
        for sym, is_dirty in dirty_state:
            elapsed = max(0.0, now - self._last_eval.get(sym, 0.0))
            rem_hb = max(0.0, self.max_interval - elapsed)
            if rem_hb < wait_sec:
                wait_sec = rem_hb
            if is_dirty:
                rem_min = max(0.0, self.min_interval - elapsed)
                if rem_min < wait_sec:
                    wait_sec = rem_min
        return wait_sec

    REGIME_CTX_MEMO_TTL_SEC = 1.0

    def _resolve_regime_context(self, symbol: str):
        sym_key = (
            self._resolve_symbol(symbol)
            if hasattr(self, "_canonical_symbols")
            else ("" if isinstance(symbol, bool) or not isinstance(symbol, str) else symbol.strip().upper())
        ) or symbol
        lock = getattr(self, "_lock", None)
        ttl = safe_nonneg_float(getattr(self, "REGIME_CTX_MEMO_TTL_SEC", 1.0), default=1.0)
        now_mono = time.monotonic()
        with lock or contextlib.nullcontext():
            memo = getattr(self, "_regime_ctx_memo", None)
            if not isinstance(memo, dict):
                memo = {}
                self._regime_ctx_memo = memo
            cached = memo.get(sym_key)
        if cached is not None:
            cached_mono, cached_ctx = cached
            if 0.0 <= (now_mono - cached_mono) < ttl:
                return cached_ctx
        kw = _provider_kw(sym_key)
        inst_mgr = getattr(self, "instant_mgr", None)
        if inst_mgr is not None:
            local_snap = resolve_runtime_market_regime_snapshot(sym_key, mgr=inst_mgr)
            if local_snap is not None:
                kw = {**kw, "snapshot": local_snap}
        try:
            regime_ctx = mkt.market_regime_context(sym_key, **kw)
        except Exception as e:
            print(f"[TrendCoordinator] Error resolving regime context for {sym_key}: {e}")
            return None
        if regime_ctx is not None and ttl > 0.0 and _parse_bool_flag(getattr(regime_ctx, "fresh", False), False):
            with lock or contextlib.nullcontext():
                memo[sym_key] = (now_mono, regime_ctx)
        return regime_ctx

    def evaluate(self, symbol):
        symbol = self._resolve_symbol(symbol)
        if symbol not in self.trend_states:
            self.trend_states[symbol] = self._new_trend_state()
        if symbol not in self.trend_states_big:
            self.trend_states_big[symbol] = self._new_trend_state()
        if not is_symbol_market_open(symbol):
            lock = getattr(self, "_lock", None)
            if lock is not None:
                with lock:
                    self._dirty[symbol], self._last_eval[symbol] = False, time.time()
            return None
        get_entry = getattr(self.current_price_mgr, "get_price", None)
        if callable(get_entry):
            entry = get_entry(symbol)
            if not entry or len(entry) < 2:
                return None
            ts_ms, current_price = entry[0], entry[1]
            ts_ms_f = safe_float(ts_ms, default=None)
            stale_fn = getattr(self.current_price_mgr, "stale_threshold_ms_for", None)
            raw_max_age_ms = (
                stale_fn(symbol)
                if callable(stale_fn)
                else getattr(self.current_price_mgr, "STALE_THRESHOLD_MS", PRICE_FRESHNESS_MAX_AGE_MS)
            )
            max_age_ms = safe_float(raw_max_age_ms, default=None)
            if ts_ms_f is None or max_age_ms is None:
                return None
            age_ms = time.time() * 1000.0 - ts_ms_f
            # get_price() attempts an HTTP refresh, but historically returned its old
            # cached entry when that refresh failed.  Do not trade on that stale value
            # or on a timestamp materially in the future.
            if age_ms > max_age_ms or age_ms < -1_000:
                print(f"[TrendCoordinator] stale/future price {symbol}: age_ms={age_ms:.0f}")
                return None
        else:
            current_price = self.current_price_mgr.get_price_value(symbol)
        current_price = safe_pos_float(current_price, default=None)
        if current_price is None:
            return None
        regime_ctx = self._resolve_regime_context(symbol)

        win = self.instant_mgr.get_window(symbol)
        win_big = self.instant_mgr.get_window(symbol, self.instant_mgr.window_big_sec)
        analyzer = self.instant_mgr.get_analyzer(symbol)
        analyzer_big = self.instant_mgr.get_analyzer(symbol, self.instant_mgr.window_big_sec)
        if win is None or win_big is None or analyzer is None or analyzer_big is None:
            return None
        kalman_fields = {}
        prev_ktrend = None
        new_ktrend = None
        if self._kalman_state is not None:
            try:
                st_prev = casefold_get(self._kalman_state._state, symbol)
                if st_prev:
                    prev_ktrend = st_prev.get("kalman_trend")
                ssot_vol = None
                inst_fname = getattr(self.instant_mgr, "filename", None)
                if isinstance(inst_fname, str) and inst_fname and os.path.exists(inst_fname):
                    try:
                        disk_st = casefold_get(read_json_dict_cached(inst_fname, default={}), symbol)
                        if isinstance(disk_st, dict):
                            ssot_vol = safe_pos_float(disk_st.get("vol_1h_pct"), default=None)
                    except Exception:
                        ssot_vol = None
                elif hasattr(self.instant_mgr, "_read_file"):
                    try:
                        disk_st = casefold_get(self.instant_mgr._read_file(), symbol)
                        if isinstance(disk_st, dict):
                            ssot_vol = safe_pos_float(disk_st.get("vol_1h_pct"), default=None)
                    except Exception:
                        ssot_vol = None
                kalman_fields = self._kalman_state.update(
                    symbol, time.time(), current_price,
                    epsilon=win.get_noise_epsilon(),
                    big_prices=win_big.prices,
                    big_sample_rate=win_big.sample_rate_sec,
                    precomputed_vol_1h_pct=ssot_vol,
                ) or {}
                new_ktrend = kalman_fields.get("kalman_trend")
            except Exception as _e:  # noqa: BLE001
                print(f"[TrendCoordinator] kalman state error {symbol} (continuing): {_e}")

        ts_small = self.trend_states[symbol]
        ts_big = self.trend_states_big[symbol]
        prev_attempts = (
            getattr(ts_small, "_last_attempt_up_ts", None),
            getattr(ts_small, "_last_attempt_down_ts", None),
            getattr(ts_big, "_last_attempt_up_ts", None),
            getattr(ts_big, "_last_attempt_down_ts", None),
        )
        prev_orders_snapshot = list(_symbol_order_ids.get(symbol, []))
        snapshot = handle_symbol(
            symbol, current_price,
            win,
            win_big,
            analyzer,
            analyzer_big,
            ts_small, ts_big,
            regime_ctx=regime_ctx,
        )
        cur_orders_snapshot = list(_symbol_order_ids.get(symbol, []))
        cur_attempts = (
            getattr(ts_small, "_last_attempt_up_ts", None),
            getattr(ts_small, "_last_attempt_down_ts", None),
            getattr(ts_big, "_last_attempt_up_ts", None),
            getattr(ts_big, "_last_attempt_down_ts", None),
        )
        handle_placed_order = bool(
            (cur_orders_snapshot and cur_orders_snapshot != prev_orders_snapshot)
            or cur_attempts != prev_attempts
        )
        with self._lock:
            self._dirty[symbol], self._last_eval[symbol] = False, time.time()
        # Merge the complete snapshot into the cross-process store. Remove ``symbol``
        # because it is already supplied positionally.
        fields = {k: v for k, v in snapshot.items() if k != "symbol"}
        if kalman_fields:
            fields.update(kalman_fields)
        if self._kalman_state is not None:
            try:
                # Primary Kalman starts an order on a trend transition for enabled
                # symbols when handle_symbol did not already place one on this tick.
                if (symbol in KALMAN_PRIMARY_SYMBOLS and prev_ktrend is not None
                        and new_ktrend != prev_ktrend and not handle_placed_order
                        and new_ktrend in (1, -1)):
                    kp_dir, kp_side, kp_reason = (
                        ("UP", "BUY", "kalman_primary_up")
                        if new_ktrend == 1
                        else ("DOWN", "SELL", "kalman_primary_down")
                    )
                    print(f"[KALMAN-PRIMARY] {symbol} ->{kp_dir}: initiating {kp_side}")
                    track_and_place_order(
                        kp_side, symbol, 1, current_price, current_price,
                        reason=kp_reason,
                        safeback_seconds=FIRE_SAFEBACK_SEC, force=False,
                        cancelorders=True, hours=0.3, regime_context=regime_ctx)
            except Exception as _e:  # noqa: BLE001
                print(f"[TrendCoordinator] kalman state error {symbol} (continuing): {_e}")
        self.instant_mgr.update_snapshot(symbol, **fields)
        return snapshot

    # Main event-driven loop and heartbeat.
    def run(self):
        touch_module_heartbeat(
            "tradeall",
            interval_sec=30.0,
            force=True,
            symbols=list(self.symbols),
            intervals={"min_interval": self.min_interval, "max_interval": self.max_interval},
        )
        while not self._stop.is_set():
            self._event.clear()
            if self._stop.is_set():
                break
            now = time.time()
            due = [s for s in self.symbols if self._is_due(s, now)]
            if due:
                touch_module_heartbeat(
                    "tradeall",
                    interval_sec=30.0,
                    symbols=list(self.symbols),
                    intervals={"min_interval": self.min_interval, "max_interval": self.max_interval},
                )
                print("----------------------------------")
                for symbol in due:
                    if self._stop.is_set():
                        break
                    try:
                        self.evaluate(symbol)
                    except Exception as e:
                        print(f"[TrendCoordinator] Error while evaluating {symbol}: {e}")
                continue
            wait_timeout = self._next_wait_timeout(time.time())
            self._event.wait(timeout=wait_timeout)
            if not self._stop.is_set():
                touch_module_heartbeat(
                    "tradeall",
                    interval_sec=30.0,
                    symbols=list(self.symbols),
                    intervals={"min_interval": self.min_interval, "max_interval": self.max_interval},
                )

    def stop(self):
        """Wake and stop the coordinator loop deterministically."""
        self._stop.set()
        self._event.set()


if __name__ == "__main__":
    _main_symbols, _, _nb_providers = tracked_price_instruments()
    _first_sym = _main_symbols[0] if _main_symbols else "BTCUSDC"
    trades = mkt.get_trades(_first_sym, 86400.0, **_provider_kw(_first_sym))
    print(f" --------- {len(trades)}")
    print(f" my trades of today : {trades}")

    # Price chain: cacheManager AF_UNIX tick stream (with 0.8s HTTP fallback if daemon absent)
    # -> CacheCurrentPriceManager (live_reader) -> Cache24PriceManager (live_reader) -> PriceWindow.
    current_price_mgr = cm.get_current_price_manager(
        symbols=_main_symbols,
        sync_ts=TIME_SLEEP_GET_PRICE,
        provider_names=_nb_providers or None,
        data_sync_mode=cm.CacheManagerInterface.MODE_LIVE_READER,
    )
    current_price_mgr.start_local_tick_stream()
    current_price_mgr.periodic_sync(TIME_SLEEP_GET_PRICE, save_state=False)
    cache24_managers = cm.CacheFactory.get(
        "24Price",
        symbols=_main_symbols,
        data_sync_mode=cm.CacheManagerInterface.MODE_LIVE_READER,
    )   # dict {symbol: Cache24PriceManager}

    # Trend manager owns windows, calculations, and live in-memory state.
    instant_mgr = cm.get_short_trend_manager(
        symbols=_main_symbols,
        data_sync_mode=cm.CacheManagerInterface.MODE_LIVE_READER,
    )
    instant_mgr.start_computation(cache24_managers, current_price_mgr)

    coordinator = TrendCoordinator(
        symbols=_main_symbols,
        instant_mgr=instant_mgr,
        current_price_mgr=current_price_mgr,
        cache24_managers=cache24_managers,
    )

    def _request_stop(_signum=None, _frame=None):
        coordinator.stop()

    _signal.signal(_signal.SIGTERM, _request_stop)
    _signal.signal(_signal.SIGINT, _request_stop)
    try:
        coordinator.run()
    finally:
        coordinator.stop()
        cm.request_stop_price_ws_streams()
        instant_mgr.shutdown(timeout=0.5)
        cm.CacheFactory.shutdown_all(timeout=0.5)
        current_price_mgr.shutdown(timeout=0.3)
        shutdown_deadline = time.monotonic() + 0.3
        cm.stop_price_ws_streams(shutdown_deadline)
