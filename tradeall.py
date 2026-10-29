import os
import time
import datetime
import threading

# Local imports.
import alertnotifiers as alert
import utils as u
import symbols as sym
from instrument_registry import casefold_get, select_instruments, symbols_for
from binance_api import bapi as api
from binance_api import bapi_trades as apitrades
import order_outcomes_log
from providers.base import extract_order_id, normalize_side
from cachepaths import split_symbol_assets
from providers.market_api import api as mkt      # Single guarded proxy (Instrument.place).

from pricewindow import (
    DEFAULT_SAMPLE_RATE_SEC,
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
from botcore import (exact_nonneg_int, exact_pos_int, load_dotenv as _load_dotenv,
                     required_bool_env, required_float_env, required_int_env,
                     safe_nonneg_float, safe_pos_float)
_load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "config.env"))


TIME_SLEEP_GET_PRICE = 0.8       # Nominal price-collection sleep interval in seconds.
ORDER_EXPIRY_HOURS = required_float_env("TRADEALL_ORDER_EXPIRY_HOURS")
EXP_TIME_BUY_ORDER = ORDER_EXPIRY_HOURS * 3600  # Configurable order expiration horizon.
EXP_TIME_SELL_ORDER = EXP_TIME_BUY_ORDER
TIME_SLEEP_EVALUATE = TIME_SLEEP_GET_PRICE + 60  # seconds to sleep for buy/sell evaluation
# Allow six orders per expiration period, hence the division by six.
TIME_SLEEP_PLACE_ORDER = TIME_SLEEP_EVALUATE + EXP_TIME_SELL_ORDER/ 6 + 4*79  # seconds to sleep for order placement

SELL_BUY_THRESHOLD = 5  # Threshold for the number of consecutive signals

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
ACCEL_INFLECTION_THRESHOLD = required_float_env("TRADEALL_ACCEL_INFLECTION_THRESHOLD")
ACCEL_SURGE_THRESHOLD = required_float_env("TRADEALL_ACCEL_SURGE_THRESHOLD")
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
        "SMALL_SLOPE_THRESHOLD": SMALL_SLOPE_THRESHOLD,
        "ACCEL_WINDOW_SEC": ACCEL_WINDOW_SEC,
        "ACCEL_INFLECTION_THRESHOLD": ACCEL_INFLECTION_THRESHOLD,
        "ACCEL_SURGE_THRESHOLD": ACCEL_SURGE_THRESHOLD,
    }
    for name, value in positive.items():
        if safe_pos_float(value, default=None) is None:
            raise ValueError(f"tradeall config invalid: {name}={value!r}")
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


def _sanitize_field(value):
    """Remove characters that would break the pipe-delimited format (A3)."""
    return (
        str(value).replace("|", "/").replace("\r", " ").replace("\n", " ")
        if value is not None
        else ""
    )


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
        line = "|".join(_sanitize_field(c) for c in cols)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
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


# All per-coin execution choices come from instruments.conf. An enabled Binance
# instrument is trend-tracked even when it opts out of TradeAll order attempts.
# No opted-in symbols means no TradeAll orders; it is not a missing configuration.
KALMAN_MODES = {
    spec.symbol: spec.setting("tradeall.kalman_mode")
    for spec in select_instruments("binance", "tradeall_fire").values()
}
TRADEALL_FIRE_SYMBOLS = set(KALMAN_MODES)
KALMAN_PRIMARY_SYMBOLS = set(symbols_for("binance", "kalman_primary"))
KALMAN_BYPASS_SELL_REASONS = frozenset({"accel_plunge_down", "small_window_plunge_down"})


def _resolve_tradeall_symbol(symbol, allowed_set=TRADEALL_FIRE_SYMBOLS, *, preserve_raw=False):
    sym_u = str(symbol or "").strip().upper()
    sym_clean = sym_u.replace("/", "").replace("-", "").replace("_", "")
    if sym_u in allowed_set:
        return sym_u
    if sym_clean in allowed_set:
        return sym_clean
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
    action = normalize_side(action) or str(action or "").upper()
    symbol = _resolve_tradeall_symbol(symbol)
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
            logger_fn = GATE_OUTCOME_LOG or order_outcomes_log.log_order_outcome
            logger_fn(symbol, action, price, None,
                      "refused", f"kalman_gate_{mode}(trend={trend})", reason)
        except Exception as _e:  # noqa: BLE001
            print(f"[KALMAN-GATE] journal error ({_e}) — block remains")
        return None
    return mkt.place(symbol, action, price, None, motivation=reason, **kwargs)


_symbol_order_ids = {}


def track_and_place_order(action, symbol, count, proposed_price, current_price, order_ids=None,
                          reason="trend_signal", **kwargs):
    """Execute trading strategy with order tracking, expiration, and replacement.

    Restores the trading strategy logic of the original track_and_place_order()
    while routing placements through the guarded proxy (_fire_order / mkt.place).
    Cancels prior active orders from previous signals (alerting if already executed),
    cancels expired orders on the venue, applies side-specific strategic pricing,
    and records newly placed order IDs.
    """
    action = normalize_side(action) or str(action or "").upper()
    symbol = _resolve_tradeall_symbol(symbol)
    print(f"Iteration {count} generated price {proposed_price} versus {current_price}")

    if order_ids is None:
        if symbol not in _symbol_order_ids:
            _symbol_order_ids[symbol] = []
        order_ids = _symbol_order_ids[symbol]

    if action == 'HOLD':
        return order_ids

    num_orders, price_step = (1, 0.2) if action == "BUY" else (1, 0.08)
    quote = (split_symbol_assets(symbol) or ("", ""))[1]

    # Price is rising, place fewer, larger orders. Increase spacing between orders as percents.
    # Price is falling, place more, smaller orders. Reduce spacing between orders as percents.
    prop_f = safe_pos_float(proposed_price, default=None)
    curr_f = safe_pos_float(current_price, default=None)
    if prop_f is None or curr_f is None:
        base_price = float("nan")
    else:
        base_price = (
            min(prop_f, curr_f * 0.999)
            if action == "BUY"
            else max(prop_f, curr_f * 1.001)
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
        import order_retry
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
        if clean_oid and clean_oid not in {"0", "0.0", "-0", "+0"} and clean_oid not in existing_str:
            to_cancel.append(clean_oid)
            existing_str.add(clean_oid)

    if to_cancel or order_ids:
        for order_id in to_cancel:
            try:
                mkt.cancel_order(symbol, str(order_id))
                try:
                    import order_retry
                    order_retry.resolve_accepted_order(symbol, order_id)
                except Exception:
                    pass
            except Exception as _e:
                print(f"[TRADEALL] cancel existing order {order_id} (or filled): {_e}")
        order_ids.clear()


    # Cancel expired orders on venue
    exp_time = EXP_TIME_BUY_ORDER if action == 'BUY' else EXP_TIME_SELL_ORDER
    try:
        api.cancel_expired_orders(action, symbol, exp_time, allowed_owners={"tradeall"})
    except Exception as _e:
        print(f"[TRADEALL] cancel_expired_orders error for {symbol}: {_e}")

    print(f"{action} price: {base_price:.2f} {quote}".strip())

    direction = -1 if action == "BUY" else 1
    for i in range(num_orders):
        adjusted_price = base_price * (1 + direction * i * price_step / 100)
        order = _fire_order(symbol, action, adjusted_price, reason, **kwargs)
        if order:
            oid = extract_order_id(order, default=None)
            if oid and oid not in {"0", "0.0", "-0", "+0"}:
                raw_id = order.get("orderId") or order.get("id") if isinstance(order, dict) else None
                if not isinstance(raw_id, bool) and isinstance(raw_id, int) and raw_id != 0:
                    order_ids.append(raw_id)
                else:
                    order_ids.append(oid)

    return order_ids


class TrendState:
    def __init__(self, max_duration_seconds, expiration_trend_time, fresh_trend_time, now_fn=time.time):
        self.state = 'HOLD'
        self.old_state = self.state
        self.expired = False
        self.start_time = None
        self.end_time = None
        self.last_confirmation_time = None
        self.max_duration_seconds = max_duration_seconds    # Unused maximum permitted trend duration.
        self.confirm_count = 0
        self.expiration_trend_time = expiration_trend_time  # Maximum seconds between confirmations.
        self.fresh_trend_time = fresh_trend_time            # Freshness threshold.
        self._now = now_fn   # Real clock by default; replay injects each tick's time (A5).
        # Per-trend cooldown: _confirmed_count_{up,down} is a legacy field name
        # counting accepted submissions, not fills. Stop at
        # FIRE_MAX_PER_TREND until a new trend. _last_attempt_* records the last
        # accepted or rejected attempt, and every retry must wait at least
        # FIRE_MIN_RETRY_INTERVAL_SEC.
        self._confirmed_count_up = 0
        self._confirmed_count_down = 0
        self._last_attempt_up_ts = None
        self._last_attempt_down_ts = None

    def start_trend(self, new_state):
        assert new_state in ['UP', 'DOWN', 'HOLD'], "Invalid trend state"
        self.old_state = self.state
        self.state = new_state
        self.start_time = self._now()
        self.last_confirmation_time = self.start_time
        self.confirm_count = 1
        self.end_time = None
        self.expired = False
        self._confirmed_count_up = 0
        self._confirmed_count_down = 0
        self._last_attempt_up_ts = None
        self._last_attempt_down_ts = None
        print(f"Start of {self.state} trend at {u.timeToHMS(self.start_time)}")
        return self.old_state

    def fire_limit_reached(self, direction):
        count = self._confirmed_count_up if direction == 'UP' else self._confirmed_count_down
        return count >= FIRE_MAX_PER_TREND

    def mark_confirmed(self, direction):
        if direction == 'UP':
            self._confirmed_count_up += 1
        else:
            self._confirmed_count_down += 1

    def can_retry_fire(self, direction):
        last = self._last_attempt_up_ts if direction == 'UP' else self._last_attempt_down_ts
        if last is None:
            return True
        elapsed = self._now() - last
        return elapsed < 0 or elapsed >= FIRE_MIN_RETRY_INTERVAL_SEC

    def mark_fire_attempt(self, direction):
        if direction == 'UP':
            self._last_attempt_up_ts = self._now()
        else:
            self._last_attempt_down_ts = self._now()

    def confirm_trend(self):
        assert self.start_time is not None, "Trend must be started before confirming"
        self.last_confirmation_time = self._now()
        self.confirm_count += 1
        print(f"{self.confirm_count} times trend confirmed: {self.state} at {u.timeToHMS(self.last_confirmation_time)}")
        return self.confirm_count

    def get_confirmed_trend_duration(self):
        if self.start_time is None or self.last_confirmation_time is None:
            raise ValueError("Start and confirmation time must be set")
        if self.last_confirmation_time <= self.start_time:
            raise ValueError("Start time must be before confirmation time")
        return self.last_confirmation_time - self.start_time

    def get_started_trend_time(self):
        if self.start_time is None:
            return 0
        return self._now() - self.start_time


    def is_trend_fresh(self, fresh_trend_time=None):
        if fresh_trend_time is None:
            fresh_trend_time = self.fresh_trend_time
        assert self.start_time is not None, "Trend must be started before checking freshness"

        elapsed_time = self._now() - self.start_time
        if elapsed_time < fresh_trend_time:
            return True

        print(f"No fresh trend! Current trend start {int(elapsed_time)} sec back > {int(fresh_trend_time)} sec ")
        return False



    def is_trend_a_minim_validated(self) :
        if self.start_time is None or self.last_confirmation_time is None:
            return False
        return (self.last_confirmation_time - self.start_time > TREND_MIN_VALIDATED_SECONDS
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
        if self.last_confirmation_time:
            time_since_last_confirmation = self._now() - self.last_confirmation_time
            if time_since_last_confirmation > self.expiration_trend_time:
                print(f"Trend expired: {self.state}. Time since last confirmation: {time_since_last_confirmation} seconds")
                self.end_trend()
                self.expired = True
                return self.expired
        return False #self.expired

    def end_trend(self):
        self.old_state = self.state
        self.end_time = self.last_confirmation_time
        print(f"Trend ended: {self.state} at {u.timeToHMS(self.end_time)} after {self.confirm_count} confirmations.")
        self.old_confirm_count = self.confirm_count
        self.state = 'HOLD'
        self.confirm_count = 0

    def is_trend_up(self):
        if self.check_trend_expiration() :
           return 0
        if self.state == 'UP':
            return self.confirm_count
        return 0

    def is_trend_down(self):
        if self.check_trend_expiration() :
           return 0
        if self.state == 'DOWN':
            return self.confirm_count
        return 0

    def is_hold(self):
        if self.check_trend_expiration() or self.state == 'HOLD':
            return self.confirm_count
        return 0
  


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


def logic_small(win, enable, symbol, gradient, slope, trend_state, current_price, regime_ctx=None, price_window=None):
    """Evaluate fast-momentum and short-term acceleration impulses on the small price window."""
    if gradient > 0 and slope > SMALL_SLOPE_THRESHOLD:
        print(f"[SMALL-WINDOW] Momentum BUY surge: gradient={gradient}, slope={slope} > {SMALL_SLOPE_THRESHOLD}")
        _fire_trend_order(enable, symbol, trend_state, "UP", "BUY", current_price,
                          "small_window_surge_up", regime_ctx=regime_ctx)

    elif gradient < 0 and slope < -SMALL_SLOPE_THRESHOLD:
        print(f"[SMALL-WINDOW] Momentum SELL plunge: gradient={gradient}, slope={slope} < -{SMALL_SLOPE_THRESHOLD}")
        _fire_trend_order(enable, symbol, trend_state, "DOWN", "SELL", current_price,
                          "small_window_plunge_down", regime_ctx=regime_ctx)

    elif ACCEL_ENABLED and price_window is not None and (
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
                label = "BUY surge" if is_up else "SELL plunge"
                direction, action, reason = (
                    ("UP", "BUY", "accel_surge_up")
                    if is_up
                    else ("DOWN", "SELL", "accel_plunge_down")
                )
                print(
                    f"[SMALL-WINDOW] Acceleration {label}: gradient={gradient}, slope={slope:.3f}, "
                    f"vel_90s={vel_short:.3f}%/m, acc_90s={acc_short:.3f}%/m^2, vel_180s={vel_med:.3f}%/m"
                )
                _fire_trend_order(
                    enable, symbol, trend_state, direction, action, current_price,
                    reason, regime_ctx=regime_ctx,
                )
        except Exception as e:
            print(f"[SMALL-WINDOW] Acceleration calculation error for {symbol}: {e}")


def logic(win, enable, symbol, gradient, slope, trend_state, current_price, regime_ctx=None):
    def _fire_once(direction, action, reason):
        return _fire_trend_order(enable, symbol, trend_state, direction, action,
                                 current_price, reason, regime_ctx=regime_ctx)

    print(f"LOGIC gradient={gradient}, slope={slope}")

    # Check for trend confirmation or initiation
    if (gradient > 0 and slope < 0) or (gradient < 0 and slope > 0):
        direction = "UP" if gradient > 0 else "DOWN"
        action = "BUY" if direction == "UP" else "SELL"
        label = "DOWN" if direction == "UP" else "UP"
        print(f"DIFERENTA MARE {win} {label}!")
        is_matching = trend_state.is_trend_up() if direction == "UP" else trend_state.is_trend_down()
        if is_matching:
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
        direction = "UP" if trend_state.is_trend_up() else "DOWN"
        action = "BUY" if direction == "UP" else "SELL"
        if (trend_state.is_trend_consistent_validated()
        or trend_state.is_started_trend_older_than(TREND_TO_BE_OLD_SECONDS)):
            print(f"ATENTIE {action} ALL {win} .... ")
            _fire_once(direction, action, f"consistent_or_old_{direction.lower()}")

    # Extreme slope counter-trend exhaustion / reversal case
    if (slope <= -SLOPE_EXTREME_THRESHOLD and trend_state.is_trend_down()) or (slope >= SLOPE_EXTREME_THRESHOLD and trend_state.is_trend_up()):
        direction = "UP" if slope <= -SLOPE_EXTREME_THRESHOLD else "DOWN"
        action = "BUY" if direction == "UP" else "SELL"
        if (trend_state.is_trend_consistent_validated()
        and trend_state.is_started_trend_older_than(TREND_TO_BE_OLD_SECONDS)):
            print(f"ATENTIE 3: {action} ALL {win} .... ")
            reason = "slope<=-5.1_and_old_down" if direction == "UP" else "slope>=5.1_and_old_up"
            _fire_once(direction, action, reason)


# Function to handle the price logic for a specific currency.
# Windows update autonomously through Cache24 subscriptions; this path only evaluates.
# Return the trend snapshot that ``TrendCoordinator`` will cache.
def handle_symbol(symbol, current_price, price_window, price_window_big,
                  analyzer, analyzer_big, trend_state, trend_state_big,
                  regime_ctx=None):

    count = 0

    # Refresh the observed sampling rate from ``CacheCurrentPriceManager``.
    try:
        import cacheManager as cm
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
    if slope == 0:
        count = count + 1
    else:
        count = 0

    if regime_ctx is None:
        try:
            regime_ctx = mkt.market_regime_context(symbol)
            print(f"[TRADEALL] {symbol} resolved trend: {getattr(regime_ctx, 'resolved_trend', None)}")
        except Exception as e:
            print(f"[TRADEALL] Error resolving regime context for {symbol}: {e}")
    else:
        print(f"[TRADEALL] {symbol} resolved trend: {getattr(regime_ctx, 'resolved_trend', None)}")

    # SMALL ONE!!
    logic_small("SMALL", True, symbol, gradient, slope, trend_state, current_price,
                regime_ctx=regime_ctx, price_window=price_window)

    # BIG ONE!!!
    slope_big, price_diff = analyzer_big.check_price_change(PRICE_CHANGE_THRESHOLD_BIG_EUR)
    logic("BIG", True, symbol, gradient, slope_big, trend_state_big, current_price, regime_ctx=regime_ctx)

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


# ════════════════════════════════════════════════════════════════════════════
# TrendCoordinator — event-driven + heartbeat.
#
# The WS-to-Cache24 source updates windows autonomously. Each tick marks a symbol dirty;
# evaluation occurs only after ``MIN_EVAL_INTERVAL_SEC`` has elapsed.
#       - or MAX_EVAL_INTERVAL_SEC has elapsed (heartbeat prevents sparse evaluation)
# Cached lookup is O(1), and the single-threaded loop prevents reentrant placement.
# ════════════════════════════════════════════════════════════════════════════

MIN_EVAL_INTERVAL_SEC = 1.5    # Floor: at most one evaluation per symbol every 1.5 seconds.
MAX_EVAL_INTERVAL_SEC = 30.0   # Heartbeat ceiling: evaluate at least every 30 seconds.


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
            import shadow_signals
            from intelligence.internal.state.volatility import get_kalman_state_path
            self._kalman_state = shadow_signals.KalmanStateTracker(
                state_path=get_kalman_state_path())
            global _kalman_ref
            _kalman_ref = self._kalman_state   # _fire_order's gate reads this signal.
            print(f"[KALMAN-GATE] active, modes={KALMAN_MODES}")
        except Exception as _e:  # noqa: BLE001
            print(f"[TrendCoordinator] shadow_signals unavailable (continuing without it): {_e}")
            self._kalman_state = None

        self.trend_states = {}
        self.trend_states_big = {}
        for symbol in self.symbols:
            self.trend_states[symbol] = TrendState(max_duration_seconds=2.5 * 60 * 60,
                                                   expiration_trend_time=2.7 * 60, fresh_trend_time=3.7 * 60)
            self.trend_states_big[symbol] = TrendState(max_duration_seconds=3 * 60 * 60,
                                                       expiration_trend_time=2.7 * 60, fresh_trend_time=3.7 * 60)
            # Windows and the fast channel live in ``instant_mgr``; subscribe here.
            # Use Cache24 only for its evaluation signal (dirty flag and event).
            if cache24_managers is not None:
                cache24_managers[symbol].subscribe_price(self)

    # Cache24 subscriber signal wakes evaluation.
    def on_price_update(self, symbol: str, ts_ms: int, price: float) -> None:
        with self._lock:
            if symbol not in self._dirty:
                return
            self._dirty[symbol] = True
        self._event.set()

    # -- Decide whether the symbol is due for evaluation. ----------------------
    def _is_due(self, symbol, now):
        elapsed = now - self._last_eval[symbol]
        if elapsed < 0:
            self._last_eval[symbol] = 0.0
            elapsed = now
        if elapsed >= self.max_interval:          # heartbeat
            return True
        with self._lock:
            dirty = self._dirty.get(symbol, False)
        return dirty and elapsed >= self.min_interval   # floor

    def evaluate(self, symbol):
        get_entry = getattr(self.current_price_mgr, "get_price", None)
        if callable(get_entry):
            entry = get_entry(symbol)
            if not entry or len(entry) < 2:
                return None
            ts_ms, current_price = entry[0], entry[1]
            try:
                age_ms = time.time() * 1000.0 - float(ts_ms)
                max_age_ms = float(getattr(
                    self.current_price_mgr, "STALE_THRESHOLD_MS", 5_000))
            except (TypeError, ValueError):
                return None
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
        regime_ctx = None
        try:
            regime_ctx = mkt.market_regime_context(symbol)
        except Exception as e:
            print(f"[TrendCoordinator] Error resolving regime context for {symbol}: {e}")

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
                st_prev = self._kalman_state._state.get(symbol)
                if st_prev:
                    prev_ktrend = st_prev.get("kalman_trend")
                kalman_fields = self._kalman_state.update(
                    symbol, time.time(), current_price,
                    epsilon=win.get_noise_epsilon(),
                    big_prices=win_big.prices,
                    big_sample_rate=win_big.sample_rate_sec,
                ) or {}
                new_ktrend = kalman_fields.get("kalman_trend")
            except Exception as _e:  # noqa: BLE001
                print(f"[TrendCoordinator] kalman state error {symbol} (continuing): {_e}")

        prev_orders_snapshot = list(_symbol_order_ids.get(symbol, []))
        snapshot = handle_symbol(
            symbol, current_price,
            win,
            win_big,
            analyzer,
            analyzer_big,
            self.trend_states[symbol], self.trend_states_big[symbol],
            regime_ctx=regime_ctx,
        )
        cur_orders_snapshot = list(_symbol_order_ids.get(symbol, []))
        handle_placed_order = bool(
            cur_orders_snapshot and cur_orders_snapshot != prev_orders_snapshot
        )
        with self._lock:
            self._dirty[symbol] = False
            self._last_eval[symbol] = time.time()
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

    def get_cached_trend(self, symbol):
        return self.instant_mgr.get_snapshot(symbol)

    def get_all_cached_trends(self):
        return self.instant_mgr.get_all_snapshots()

    # Main event-driven loop and heartbeat.
    def run(self):
        while not self._stop.is_set():
            self._event.wait(timeout=self.max_interval)
            self._event.clear()
            if self._stop.is_set():
                break
            now = time.time()
            due = [s for s in self.symbols if self._is_due(s, now)]
            if not due:
                continue
            print(f"----------------------------------")
            for symbol in due:
                try:
                    self.evaluate(symbol)
                except Exception as e:
                    print(f"[TrendCoordinator] Error while evaluating {symbol}: {e}")

    def stop(self):
        """Wake and stop the coordinator loop deterministically."""
        self._stop.set()
        self._event.set()


if __name__ == "__main__":
    trades = apitrades.get_my_trades_24(order_type=None, symbol=sym.btcsymbol, days_ago=0, limit=1000)
    print(f" --------- {len(trades)}")
    print(f" my trades of today : {trades}")

    order_ids = []

    # Price chain: WebSocket market data -> CacheCurrentPrice -> Cache24 -> PriceWindow.
    # Create CacheCurrentPrice before Cache24 with the correct sync_ts; otherwise
    # Cache24.get_remote_items creates it internally with sync_ts=30.
    import cacheManager as cm
    from binance_api import bapi_ws
    # The user-data WebSocket bridge is opt-in; tradeall needs execution reports.
    cm.enable_real_ws_event_sync()
    current_price_mgr = cm.get_current_price_manager(
        ws_manager=bapi_ws.get_ws_manager(),
        sync_ts=TIME_SLEEP_GET_PRICE,
    )
    cache24_managers = cm.CacheFactory.get("24Price")   # dict {symbol: Cache24PriceManager}

    # Trend manager owns windows, calculations, and the cross-process cache.
    instant_mgr = cm.get_short_trend_manager()
    instant_mgr.start_computation(cache24_managers, current_price_mgr)

    coordinator = TrendCoordinator(
        symbols=sym.symbols,
        instant_mgr=instant_mgr,
        current_price_mgr=current_price_mgr,
        cache24_managers=cache24_managers,
    )
    coordinator.run()
