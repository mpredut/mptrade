from __future__ import annotations
import contextlib
import copy
import os
import hashlib
import math
import signal
import statistics
import threading
import time

import numpy as np

from providers.order_id_context import (
    can_bot_preempt_order,
    extract_client_order_id,
    resolve_priority_bot_owner,
)
from providers.market_api import api as mkt   # Single guarded proxy (Instrument.place).
api = mkt
from providers.execution_audit import ExecutionAudit
from providers.order_ohlc_utils import extract_remaining_qty
from order_lifecycle import (
    has_preemptible_open_orders,
    preempt_lower_priority_open_orders,
    propagate_submission_refusal,
    restore_cancelled_instrument_orders,
)
from order_retry import resolve_accepted_order
from providers.quantity import resolve_assets
from state_io import load_json_state, read_json_dict, save_json_state
from guard_hierarchy import GUARD_REGISTRY
from intelligence.macro_sentiment_ssot import load_cached_sentiment_advisor_eval
import order_guard as _og
import pricewindow as pw

from cacheInterface import CacheManagerInterface
from cacheaccount import get_asset_value_manager
from cacheprice import (
    get_24_price_manager,
    request_stop_price_ws_streams,
    stop_price_ws_streams,
)
import cachetrend as cm
from instrument_registry import casefold_get, symbols_for

# Load tunable parameters from the versioned, secret-free config before reading
# environment variables below. load_dotenv does not overwrite real environment.
from botcore import (
    FileLock,
    _parse_bool_flag,
    clamp_float,
    coerce_finite_float,
    exact_nonneg_int,
    extract_order_id,
    extract_order_price,
    extract_order_side,
    extract_timestamp_s,
    load_root_config,
    normalize_epoch_sec,
    require_int,
    require_pos_float,
    required_float_env,
    required_int_env,
    safe_nonneg_float,
    safe_pos_float,
    safe_pos_int,
    touch_module_heartbeat,
    validate_required_env_keys,
)
from cachepaths import ASSETGUARDIAN_STATE_FILENAME, cache_path, resolve_cachedb_filepath
load_root_config()

REQUIRED_CONFIG_KEYS = (
    "AG_CHECK_INTERVAL_SEC", "AG_REFERENCE_MINUTES_BACK", "AG_BUY_USE_CASH_RATIO",
    "AG_SELL_REARM_GROWTH_PCT", "AG_ORDER_MAX_AGE_SEC", "AG_RECOVERY_RESET_PCT",
    "AG_NEAR_TRIGGER_SEC", "AG_ACTIVE_TRIGGER_SEC", "AG_NEAR_TRIGGER_DISTANCE_PCT",
    "AG_TREND_DEFER_MAX_SEC", "AG_ORDER_MISSING_CONFIRMATIONS", "AG_PORTFOLIO_GROWTH_PCT",
    "AG_PORTFOLIO_ASSET_MIN_GROWTH_PCT", "AG_PORTFOLIO_SELL_ALLOCATION",
    "AG_PORTFOLIO_SELL_MAX_DISCOUNT", "AG_RTRADE_BLOCK_CANCEL_SEC", "AG_P3B_MAX_AGE_SEC",
    "AG_P3B_SELL_DEFENSIVE_MAX_DISCOUNT", "AG_P3B_SELL_TRIM_MAX_DISCOUNT",
    "AG_P3B_BUY_ACCUMULATE_MAX_DISCOUNT", "AG_P3B_BUY_DEFENSIVE_MAX_PENALTY",
)


# Every financial/operational parameter is mandatory in config.env.
# There are deliberately no hidden code defaults: a missing key stops startup.
validate_required_env_keys(REQUIRED_CONFIG_KEYS, label="AssetGuardian")
CHECK_INTERVAL_SECONDS = required_float_env("AG_CHECK_INTERVAL_SEC")
ASSET_REFERENCE_MINUTES_BACK_DEFAULT = required_float_env("AG_REFERENCE_MINUTES_BACK")
BUY_USE_CASH_RATIO = required_float_env("AG_BUY_USE_CASH_RATIO")
SELL_REARM_GROWTH_PERCENT = required_float_env("AG_SELL_REARM_GROWTH_PCT")
ORDER_MAX_AGE_SECONDS = required_float_env("AG_ORDER_MAX_AGE_SEC")
RECOVERY_RESET_PERCENT = required_float_env("AG_RECOVERY_RESET_PCT")
NEAR_TRIGGER_SECONDS = required_float_env("AG_NEAR_TRIGGER_SEC")
ACTIVE_TRIGGER_SECONDS = required_float_env("AG_ACTIVE_TRIGGER_SEC")
NEAR_TRIGGER_DISTANCE_PCT = required_float_env("AG_NEAR_TRIGGER_DISTANCE_PCT")
TREND_DEFER_MAX_SECONDS = required_float_env("AG_TREND_DEFER_MAX_SEC")
PORTFOLIO_GROWTH_PERCENT = required_float_env("AG_PORTFOLIO_GROWTH_PCT")
PORTFOLIO_ASSET_MIN_GROWTH_PERCENT = required_float_env("AG_PORTFOLIO_ASSET_MIN_GROWTH_PCT")
PORTFOLIO_SELL_ALLOCATION = required_float_env("AG_PORTFOLIO_SELL_ALLOCATION")
PORTFOLIO_SELL_MAX_DISCOUNT = required_float_env("AG_PORTFOLIO_SELL_MAX_DISCOUNT")
RTRADE_BLOCK_CANCEL_SECONDS = required_float_env("AG_RTRADE_BLOCK_CANCEL_SEC")
P3B_MAX_AGE_SECONDS = required_float_env("AG_P3B_MAX_AGE_SEC")
P3B_SELL_DEFENSIVE_MAX_DISCOUNT = required_float_env("AG_P3B_SELL_DEFENSIVE_MAX_DISCOUNT")
P3B_SELL_TRIM_MAX_DISCOUNT = required_float_env("AG_P3B_SELL_TRIM_MAX_DISCOUNT")
P3B_BUY_ACCUMULATE_MAX_DISCOUNT = required_float_env("AG_P3B_BUY_ACCUMULATE_MAX_DISCOUNT")
P3B_BUY_DEFENSIVE_MAX_PENALTY = required_float_env("AG_P3B_BUY_DEFENSIVE_MAX_PENALTY")


def _progressive_tier_allocations(count, initial_allocation, *, mode="convex"):
    """Build monotonically increasing tier allocations starting at ``initial_allocation``."""
    n = require_int(count, "tier count", min_value=1, error_msg="tier count must be >= 1")
    if n == 1:
        return (1.0,)
    w0 = safe_pos_float(initial_allocation, default=None)
    if w0 is None or w0 * n > 1.0 + 1e-12:
        raise ValueError(f"initial_allocation={initial_allocation!r} must be in (0, 1/{n}]")
    surplus = max(0.0, 1.0 - w0 * n)
    if mode == "convex":
        deltas = [i * (i + 1) / 2.0 for i in range(n)]
    elif mode == "linear":
        deltas = [i * 1.0 for i in range(n)]
    else:
        raise ValueError(f"unsupported progressive allocation mode: {mode!r}")
    delta_sum = sum(deltas)
    if delta_sum <= 0:
        weights = [round(1.0 / n, 6) for _ in range(n - 1)]
    else:
        weights = [round(w0 + surplus * (deltas[i] / delta_sum), 6) for i in range(n - 1)]
    weights.append(round(1.0 - sum(weights), 6))
    return tuple(weights)


def _build_dynamic_buy_tiers(thresholds=None, initial_allocation=None):
    """Derive BUY drawdown thresholds and pyramid allocations dynamically from config bounds."""
    if thresholds is None:
        count = max(2, round((1.0 - P3B_BUY_ACCUMULATE_MAX_DISCOUNT) / max(1e-6, PORTFOLIO_SELL_ALLOCATION)))
        discount_floor = math.ceil(
            (RECOVERY_RESET_PERCENT + 0.5 * NEAR_TRIGGER_DISTANCE_PCT)
            / max(1e-6, 1.0 - P3B_BUY_ACCUMULATE_MAX_DISCOUNT)
        )
        first_thr = max(RECOVERY_RESET_PERCENT + 2.0 * NEAR_TRIGGER_DISTANCE_PCT, discount_floor * 1.0)
        computed = [round(first_thr, 4)]
        for idx in range(1, count):
            step = RECOVERY_RESET_PERCENT + (idx - 1) * (0.5 * NEAR_TRIGGER_DISTANCE_PCT)
            computed.append(round(computed[-1] + step, 4))
        ordered = tuple(computed)
    else:
        ordered = tuple(sorted(require_pos_float(t, "threshold") for t in thresholds))
    if initial_allocation is None:
        initial_allocation = min(PORTFOLIO_SELL_ALLOCATION, 1.0 / len(ordered))
    return tuple(zip(ordered, _progressive_tier_allocations(len(ordered), initial_allocation, mode="convex")))


BUY_TIERS = _build_dynamic_buy_tiers()
_SELL_COUNT = max(2, round(1.0 / max(1e-6, PORTFOLIO_SELL_ALLOCATION)))
_SELL_CORE_COUNT = max(2, round((1.0 - P3B_BUY_ACCUMULATE_MAX_DISCOUNT) / max(1e-6, PORTFOLIO_SELL_ALLOCATION)))
_SELL_FIRST_THR = (
    math.ceil(PORTFOLIO_ASSET_MIN_GROWTH_PERCENT / max(1e-6, 1.0 - P3B_SELL_DEFENSIVE_MAX_DISCOUNT))
    * 1.0
)
_SELL_BREAKOUT_STEP = _SELL_FIRST_THR + 0.5 * NEAR_TRIGGER_DISTANCE_PCT
_SELL_COMPUTED = [round(_SELL_FIRST_THR, 4)]
for _sell_idx in range(1, _SELL_COUNT):
    _sell_step = RECOVERY_RESET_PERCENT if _sell_idx < _SELL_CORE_COUNT else _SELL_BREAKOUT_STEP
    _SELL_COMPUTED.append(round(_SELL_COMPUTED[-1] + _sell_step, 4))
_SELL_ORDERED = tuple(_SELL_COMPUTED)
_SELL_INIT_ALLOC = round(
    (1.0 / len(_SELL_ORDERED)) * (1.0 - ((len(_SELL_ORDERED) - 1) / 2.0) * PORTFOLIO_SELL_MAX_DISCOUNT), 6
)
SELL_TIERS = tuple(
    zip(_SELL_ORDERED, _progressive_tier_allocations(len(_SELL_ORDERED), _SELL_INIT_ALLOC, mode="linear"))
)
GUARDED_BUY_THRESHOLD = min((t for t, _ in BUY_TIERS), default=None)
TRACKED_SYMBOLS = tuple(symbols_for("binance", "assetguardian"))
LEGACY_BUY_SYMBOL = TRACKED_SYMBOLS[0] if TRACKED_SYMBOLS else "BTCUSDC"
ORDER_MISSING_CONFIRMATIONS = required_int_env("AG_ORDER_MISSING_CONFIRMATIONS")


class AssetGuardianState:
    def __init__(self, path=None, *, fail_closed=True):
        self.path = path or cache_path(ASSETGUARDIAN_STATE_FILENAME)
        self.lock_path = self.path + ".lock"
        self.fail_closed = _parse_bool_flag(fail_closed, default=True)

    def _resolved_paths(self, *, migrate: bool = False) -> tuple[str, str]:
        target_path = str(resolve_cachedb_filepath(self.path, migrate=migrate))
        lock_path = (
            f"{target_path}.lock"
            if self.lock_path == f"{self.path}.lock"
            else str(resolve_cachedb_filepath(self.lock_path, migrate=migrate))
        )
        return target_path, lock_path

    def load(self):
        target_path, lock_path = self._resolved_paths(migrate=True)
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        with FileLock(lock_path):
            return load_json_state(
                target_path, default_factory=dict, fail_closed=self.fail_closed, label="AssetGuardian"
            )

    def save(self, value):
        target_path, lock_path = self._resolved_paths(migrate=True)
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        with FileLock(lock_path):
            save_json_state(target_path, value, sort_keys=True, separators=(",", ":"))


STATE = AssetGuardianState()
_CYCLE_SCOPE = threading.local()
_last_evaluation = {
    symbol: {"growth": None, "drawdown": None, "pending_tier": False, "volatility_mult": 1.0}
    for symbol in TRACKED_SYMBOLS
}
_local_price_samples = {symbol: [] for symbol in TRACKED_SYMBOLS}


@contextlib.contextmanager
def _cycle_io_scope():
    """Reuse normalized state root and global telemetry within one evaluation pass."""
    existing = getattr(_CYCLE_SCOPE, "cache", None)
    if existing is not None:
        yield existing
        return
    cache = {
        "state_id": None,
        "state_root": None,
        "portfolio_growth": {},
        "p3b_loaded": False,
        "p3b_val": None,
    }
    _CYCLE_SCOPE.cache = cache
    try:
        yield cache
    finally:
        _CYCLE_SCOPE.cache = None


def _validate_tier_table(tiers, name: str, *, exact_sum: bool = False):
    if not tiers or any(
        safe_pos_float(threshold, default=None) is None or safe_pos_float(allocation, default=None) is None
        for threshold, allocation in tiers
    ):
        raise ValueError(f"{name} must contain positive threshold:allocation pairs")
    if len({threshold for threshold, _ in tiers}) != len(tiers):
        raise ValueError(f"{name} contains duplicate thresholds")
    total_w = sum(allocation for _, allocation in tiers)
    if exact_sum and abs(total_w - 1.0) > 1e-12:
        raise ValueError(f"the {name} weights must sum to exactly 1")
    if not exact_sum and total_w > 1.0 + 1e-12:
        raise ValueError(f"the {name} weights sum exceeds 1")


if CHECK_INTERVAL_SECONDS <= 0:
    raise ValueError("AG_CHECK_INTERVAL_SEC must be > 0")
if ASSET_REFERENCE_MINUTES_BACK_DEFAULT <= 0:
    raise ValueError("AG_REFERENCE_MINUTES_BACK must be > 0")
if not 0 < BUY_USE_CASH_RATIO <= 1:
    raise ValueError("AG_BUY_USE_CASH_RATIO must be in (0, 1]")
_validate_tier_table(BUY_TIERS, "AG_BUY_TIERS", exact_sum=False)
_validate_tier_table(SELL_TIERS, "AG_SELL_TIERS", exact_sum=True)
if SELL_REARM_GROWTH_PERCENT <= 0 or SELL_REARM_GROWTH_PERCENT >= SELL_TIERS[0][0]:
    raise ValueError("AG_SELL_REARM_GROWTH_PCT must be > 0 and below the first SELL threshold")
if ORDER_MISSING_CONFIRMATIONS <= 0:
    raise ValueError("AG_ORDER_MISSING_CONFIRMATIONS must be an integer and > 0")
for _ag_val, _ag_name in (
    (ORDER_MAX_AGE_SECONDS, "AG_ORDER_MAX_AGE_SEC"),
    (RECOVERY_RESET_PERCENT, "AG_RECOVERY_RESET_PCT"),
    (NEAR_TRIGGER_SECONDS, "AG_NEAR_TRIGGER_SEC"),
    (ACTIVE_TRIGGER_SECONDS, "AG_ACTIVE_TRIGGER_SEC"),
    (NEAR_TRIGGER_DISTANCE_PCT, "AG_NEAR_TRIGGER_DISTANCE_PCT"),
    (TREND_DEFER_MAX_SECONDS, "AG_TREND_DEFER_MAX_SEC"),
    (PORTFOLIO_GROWTH_PERCENT, "AG_PORTFOLIO_GROWTH_PCT"),
    (PORTFOLIO_ASSET_MIN_GROWTH_PERCENT, "AG_PORTFOLIO_ASSET_MIN_GROWTH_PCT"),
    (RTRADE_BLOCK_CANCEL_SECONDS, "AG_RTRADE_BLOCK_CANCEL_SEC"),
    (P3B_MAX_AGE_SECONDS, "AG_P3B_MAX_AGE_SEC"),
):
    if _ag_val <= 0:
        raise ValueError(f"{_ag_name} must be finite and > 0")
for _ag_val, _ag_hi, _ag_name in (
    (PORTFOLIO_SELL_ALLOCATION, 1.0, "AG_PORTFOLIO_SELL_ALLOCATION"),
    (PORTFOLIO_SELL_MAX_DISCOUNT, 0.50, "AG_PORTFOLIO_SELL_MAX_DISCOUNT"),
    (P3B_SELL_DEFENSIVE_MAX_DISCOUNT, 0.50, "AG_P3B_SELL_DEFENSIVE_MAX_DISCOUNT"),
    (P3B_BUY_ACCUMULATE_MAX_DISCOUNT, 0.50, "AG_P3B_BUY_ACCUMULATE_MAX_DISCOUNT"),
    (P3B_BUY_DEFENSIVE_MAX_PENALTY, 1.0, "AG_P3B_BUY_DEFENSIVE_MAX_PENALTY"),
):
    if not 0 < _ag_val <= _ag_hi:
        raise ValueError(f"{_ag_name} must be in (0, {_ag_hi:.2f}]" if _ag_hi < 1.0 else f"{_ag_name} must be in (0, 1.0]")
del _ag_val, _ag_hi, _ag_name
if PORTFOLIO_ASSET_MIN_GROWTH_PERCENT <= SELL_REARM_GROWTH_PERCENT:
    raise ValueError("AG_PORTFOLIO_ASSET_MIN_GROWTH_PCT must be above AG_SELL_REARM_GROWTH_PCT")
if not 0 < P3B_SELL_TRIM_MAX_DISCOUNT <= P3B_SELL_DEFENSIVE_MAX_DISCOUNT:
    raise ValueError("AG_P3B_SELL_TRIM_MAX_DISCOUNT must be in (0, AG_P3B_SELL_DEFENSIVE_MAX_DISCOUNT]")
if SELL_TIERS[0][0] * (1.0 - P3B_SELL_DEFENSIVE_MAX_DISCOUNT) <= SELL_REARM_GROWTH_PERCENT:
    raise ValueError("Compressed first SELL tier must remain above AG_SELL_REARM_GROWTH_PCT")
if BUY_TIERS[0][0] * (1.0 - P3B_BUY_ACCUMULATE_MAX_DISCOUNT) <= RECOVERY_RESET_PERCENT:
    raise ValueError("Compressed first BUY tier must remain above AG_RECOVERY_RESET_PCT")


ORDER_LIFECYCLE = mkt.tracked_order_lifecycle(
    provider_name="binance",
    venue="Binance",
    missing_confirmations=ORDER_MISSING_CONFIRMATIONS,
    retry_on_lookup_error=True,
    max_age_seconds=ORDER_MAX_AGE_SECONDS,
    audit=ExecutionAudit(),
    clock=lambda: time.time(),
)


_finite_float = coerce_finite_float


def _price_row(row):
    """Normalize a Price24 row to ``(timestamp_seconds, positive_price)``."""
    if isinstance(row, dict):
        timestamp = extract_timestamp_s(row, default=None)
        price = extract_order_price(row, default=None)
        return (timestamp, price) if (timestamp is not None and price is not None) else None
    if not isinstance(row, (list, tuple)) or len(row) < 2:
        return None
    timestamp = normalize_epoch_sec(row[0], default=None)
    price = safe_pos_float(row[1], default=None)
    return (timestamp, price) if (timestamp is not None and price is not None) else None


def _read_symbol_price_rows(symbol):
    """Copy one symbol's shared 24h price history under the manager lock."""
    try:
        managers = get_24_price_manager(symbols=list(TRACKED_SYMBOLS))
        manager = casefold_get(managers, symbol) if isinstance(managers, dict) else None
        if manager is None:
            print(f"ERROR 24Price manager missing for {symbol}")
            return []
        if hasattr(manager, "resync_mem_file"):
            manager.resync_mem_file()
        with manager.lock:
            rows = list(casefold_get(manager.cache, symbol, []))
        print(f"[DEBUG] {symbol} 24Price rows loaded: {len(rows)}")
        return rows
    except Exception as e:
        print(f"ERROR reading 24Price cache for {symbol}: {e}")
        return []


def _volatility_mult_bounds():
    """Derive symmetric volatility multiplier bounds from configured discount parameters."""
    max_swing = min(0.45, PORTFOLIO_SELL_MAX_DISCOUNT * (1.0 + PORTFOLIO_SELL_MAX_DISCOUNT))
    return round(1.0 - max_swing, 4), round(1.0 + max_swing, 4)


def _compute_symbol_volatility_multiplier(
    normalized_rows, *, bucket_sec=None, ref_step_pct=None, min_mult=None, max_mult=None,
):
    """Compute a bounded per-symbol 24h volatility multiplier around 1.0."""
    if bucket_sec is None:
        bucket_sec = max(60.0, (safe_pos_float(ORDER_MAX_AGE_SECONDS, default=60.0) or 60.0))
    if ref_step_pct is None:
        ref_min_back = safe_pos_float(ASSET_REFERENCE_MINUTES_BACK_DEFAULT, default=1440.0) or 1440.0
        buckets_per_window = max(1.0, (ref_min_back * 60.0) / bucket_sec)
        ref_step_pct = (RECOVERY_RESET_PERCENT + 0.5 * NEAR_TRIGGER_DISTANCE_PCT) / math.sqrt(buckets_per_window)
    default_min, default_max = _volatility_mult_bounds()
    min_mult = default_min if min_mult is None else min_mult
    max_mult = default_max if max_mult is None else max_mult
    if not isinstance(normalized_rows, (list, tuple)) or len(normalized_rows) < 60:
        return 1.0
    ordered = sorted(
        (
            (ts_f, px_f)
            for r in normalized_rows
            if isinstance(r, dict)
            and (ts_f := safe_pos_float(r.get("timestamp"), default=None)) is not None
            and (px_f := safe_pos_float(r.get("price"), default=None)) is not None
        ),
        key=lambda item: item[0],
    )
    if len(ordered) < 60 or (ordered[-1][0] - ordered[0][0]) < 8.0 * bucket_sec:
        return 1.0
    buckets = {math.trunc(ts // bucket_sec): px for ts, px in ordered}
    bucket_prices = [buckets[k] for k in sorted(buckets)]
    if len(bucket_prices) < 8:
        return 1.0
    abs_returns = [
        abs((curr - prev) / prev) * 100.0
        for prev, curr in zip(bucket_prices, bucket_prices[1:])
        if prev > 0
    ]
    if not abs_returns:
        return 1.0
    mean_abs_ret_pct = sum(abs_returns) / len(abs_returns)
    if mean_abs_ret_pct <= 0 or ref_step_pct <= 0:
        return 1.0
    return round(clamp_float((mean_abs_ret_pct / ref_step_pct) ** 0.5, min_mult, max_mult, default=1.0), 4)


def _clamp_volatility_mult(val) -> float:
    min_mult, max_mult = _volatility_mult_bounds()
    return clamp_float(safe_pos_float(val, default=1.0), min_mult, max_mult, default=1.0)


def _symbol_volatility_mult(symbol):
    """Return the latest bounded 24h volatility multiplier for ``symbol`` (default 1.0)."""
    row = casefold_get(_last_evaluation, symbol, {})
    return _clamp_volatility_mult(row.get("volatility_mult") if isinstance(row, dict) else None)


def _log_symbol_and_return(symbol: str, message: str, ret_val=False):
    print(f"[{symbol}] {message}")
    return ret_val


_WINDOW_OUTLIER_NEIGHBOR_AGREE_RATIO = 1.20
_WINDOW_OUTLIER_SPIKE_RATIO = 1.35
_WINDOW_OUTLIER_MEDIAN_RATIO = 2.5
_WINDOW_OUTLIER_CORROBORATE_RATIO = 1.25
_CORRUPT_PEAK_IN_WINDOW_RATIO = 1.5
_CORRUPT_PEAK_EXTREME_RATIO = 10.0


def _filter_isolated_price_outliers(rows: list[dict]) -> tuple[list[dict], int]:
    """Remove isolated single-tick spikes and uncorroborated extreme outliers from rolling prices."""
    if len(rows) < 3:
        return list(rows), 0
    ordered = sorted(rows, key=lambda row: row["timestamp"])
    n = len(ordered)
    drop_idx: set[int] = set()
    for i in range(1, n - 1):
        prev_px = ordered[i - 1]["price"]
        cur_px = ordered[i]["price"]
        next_px = ordered[i + 1]["price"]
        lo_nb = min(prev_px, next_px)
        hi_nb = max(prev_px, next_px)
        if lo_nb > 0.0 and (hi_nb / lo_nb) <= _WINDOW_OUTLIER_NEIGHBOR_AGREE_RATIO:
            if cur_px > hi_nb * _WINDOW_OUTLIER_SPIKE_RATIO or cur_px < lo_nb / _WINDOW_OUTLIER_SPIKE_RATIO:
                drop_idx.add(i)
    kept = [row for i, row in enumerate(ordered) if i not in drop_idx]
    removed = len(drop_idx)
    if len(kept) >= 5:
        med_px = statistics.median(row["price"] for row in kept)
        if med_px > 0.0:
            extreme_indices = [
                i for i, row in enumerate(kept)
                if max(row["price"] / med_px, med_px / row["price"]) > _WINDOW_OUTLIER_MEDIAN_RATIO
            ]
            if extreme_indices:
                final_drop: set[int] = set()
                for idx in extreme_indices:
                    px = kept[idx]["price"]
                    corroborated = 0
                    for j, other in enumerate(kept):
                        if j == idx:
                            continue
                        o_px = other["price"]
                        if max(o_px / px, px / o_px) <= _WINDOW_OUTLIER_CORROBORATE_RATIO:
                            corroborated += 1
                            if corroborated >= 2:
                                break
                    if corroborated < 2:
                        final_drop.add(idx)
                if final_drop:
                    kept = [row for i, row in enumerate(kept) if i not in final_drop]
                    removed += len(final_drop)
    return kept, removed


def _is_corrupted_buy_peak(
    stored_peak,
    stored_peak_ts,
    window_max_price,
    current_price=None,
    *,
    window_start_ts=None,
    now_ts=None,
) -> bool:
    """Return True when a persisted BUY campaign peak_price is clearly contaminated."""
    sp = _finite_float(stored_peak, positive=True)
    wm = _finite_float(window_max_price, positive=True)
    if sp is None or wm is None:
        return False
    cp = _finite_float(current_price, positive=True)
    anchor = max(wm, cp) if cp is not None else wm
    if sp > anchor * _CORRUPT_PEAK_EXTREME_RATIO:
        return True
    pts = _finite_float(stored_peak_ts, positive=True)
    w_start = _finite_float(window_start_ts, positive=True)
    eff_now = _finite_float(now_ts, positive=True) or (time.time() * 1.0)
    if pts is not None and w_start is not None and w_start <= pts <= eff_now + 60.0:
        if sp > wm * _CORRUPT_PEAK_IN_WINDOW_RATIO:
            return True
    return False


def _is_corrupted_sell_trough(
    stored_trough,
    stored_trough_ts,
    window_min_price,
    current_price=None,
    *,
    window_start_ts=None,
    now_ts=None,
) -> bool:
    """Return True when a persisted SELL campaign trough_price is clearly contaminated."""
    st = _finite_float(stored_trough, positive=True)
    wmin = _finite_float(window_min_price, positive=True)
    if st is None or wmin is None:
        return False
    cp = _finite_float(current_price, positive=True)
    lo_anchor = min(wmin, cp) if cp is not None else wmin
    hi_anchor = max(wmin, cp) if cp is not None else wmin
    if st < lo_anchor / _CORRUPT_PEAK_EXTREME_RATIO or st > hi_anchor * _CORRUPT_PEAK_EXTREME_RATIO:
        return True
    tts = _finite_float(stored_trough_ts, positive=True)
    w_start = _finite_float(window_start_ts, positive=True)
    eff_now = _finite_float(now_ts, positive=True) or (time.time() * 1.0)
    if tts is not None and w_start is not None and w_start <= tts <= eff_now + 60.0:
        if st < wmin / _CORRUPT_PEAK_IN_WINDOW_RATIO:
            return True
    return False


def _compute_symbol_volatility_multiplier_arrays(
    ts_arr: np.ndarray,
    px_arr: np.ndarray,
    *,
    bucket_sec=None,
    ref_step_pct=None,
    min_mult=None,
    max_mult=None,
) -> float:
    """Vectorized equivalent of ``_compute_symbol_volatility_multiplier`` for sorted ``(ts_arr, px_arr)``."""
    if bucket_sec is None:
        bucket_sec = max(60.0, (safe_pos_float(ORDER_MAX_AGE_SECONDS, default=60.0) or 60.0))
    if ref_step_pct is None:
        ref_min_back = safe_pos_float(ASSET_REFERENCE_MINUTES_BACK_DEFAULT, default=1440.0) or 1440.0
        buckets_per_window = max(1.0, (ref_min_back * 60.0) / bucket_sec)
        ref_step_pct = (RECOVERY_RESET_PERCENT + 0.5 * NEAR_TRIGGER_DISTANCE_PCT) / math.sqrt(buckets_per_window)
    default_min, default_max = _volatility_mult_bounds()
    min_mult = default_min if min_mult is None else min_mult
    max_mult = default_max if max_mult is None else max_mult
    n = ts_arr.size
    if n < 60:
        return 1.0
    if not bool(np.all(ts_arr[:-1] <= ts_arr[1:])):
        order = np.argsort(ts_arr, kind="mergesort")
        ts_arr = ts_arr[order]
        px_arr = px_arr[order]
    if (ts_arr[-1] - ts_arr[0]).item() < 8.0 * bucket_sec:
        return 1.0
    bucket_ids = np.floor(ts_arr / bucket_sec).astype(np.int64)
    change_idx = np.flatnonzero(bucket_ids[:-1] != bucket_ids[1:])
    last_idx = np.empty(change_idx.size + 1, dtype=np.intp)
    if change_idx.size:
        last_idx[:-1] = change_idx
    last_idx[-1] = n - 1
    if last_idx.size < 8:
        return 1.0
    bp = px_arr[last_idx]
    prev_bp = bp[:-1]
    curr_bp = bp[1:]
    pos_mask = prev_bp > 0.0
    if not bool(np.any(pos_mask)):
        return 1.0
    abs_returns = np.abs((curr_bp[pos_mask] - prev_bp[pos_mask]) / prev_bp[pos_mask]) * 100.0
    mean_abs_ret_pct = np.mean(abs_returns).item()
    if mean_abs_ret_pct <= 0.0 or ref_step_pct <= 0.0:
        return 1.0
    return round(clamp_float((mean_abs_ret_pct / ref_step_pct) ** 0.5, min_mult, max_mult, default=1.0), 4)


def _extract_window_arrays(rows, target_ts: float, now_ts: float) -> tuple[np.ndarray, np.ndarray]:
    """Extract in-window ``(timestamp_sec, price)`` arrays without per-row dict allocation."""
    ts_list: list[float] = []
    px_list: list[float] = []
    isfinite = math.isfinite
    for raw in rows:
        t_raw = type(raw)
        if t_raw is list or t_raw is tuple:
            if len(raw) >= 2:
                r0 = raw[0]
                r1 = raw[1]
                t0 = type(r0)
                t1 = type(r1)
                if (t0 is int or t0 is float) and (t1 is int or t1 is float):
                    if r0 > 0 and r1 > 0 and isfinite(r0) and isfinite(r1):
                        ts_val = (r0 / 1000.0) if r0 >= 1e11 else (r0 * 1.0)
                        if target_ts <= ts_val <= now_ts:
                            ts_list.append(ts_val)
                            px_list.append(r1 * 1.0)
                    continue
        parsed = _price_row(raw)
        if parsed is None:
            continue
        ts_val, px_val = parsed
        if target_ts <= ts_val <= now_ts:
            ts_list.append(ts_val)
            px_list.append(px_val)
    if not ts_list:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty
    return np.asarray(ts_list, dtype=np.float64), np.asarray(px_list, dtype=np.float64)


def _sort_and_dedup_ts_px(ts_arr: np.ndarray, px_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return timestamp-sorted arrays with exact ``(ts, px)`` duplicates removed."""
    n = ts_arr.size
    if n <= 1:
        return ts_arr, px_arr
    if not bool(np.all(ts_arr[:-1] <= ts_arr[1:])):
        order = np.argsort(ts_arr, kind="mergesort")
        ts_arr = ts_arr[order]
        px_arr = px_arr[order]
    same_ts = ts_arr[1:] == ts_arr[:-1]
    if not bool(np.any(same_ts)):
        return ts_arr, px_arr
    keep = np.ones(n, dtype=bool)
    i = 0
    while i < n:
        j = i + 1
        if j < n and ts_arr[j] == ts_arr[i]:
            seen_px = {px_arr[i].item()}
            while j < n and ts_arr[j] == ts_arr[i]:
                pv = px_arr[j].item()
                if pv in seen_px:
                    keep[j] = False
                else:
                    seen_px.add(pv)
                j += 1
        i = j
    if bool(np.all(keep)):
        return ts_arr, px_arr
    return ts_arr[keep], px_arr[keep]


def _filter_isolated_price_outliers_arrays(
    ts_arr: np.ndarray, px_arr: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """Vectorized equivalent of ``_filter_isolated_price_outliers`` on sorted ``(ts_arr, px_arr)``."""
    n = ts_arr.size
    if n < 3:
        return ts_arr, px_arr, 0
    if not bool(np.all(ts_arr[:-1] <= ts_arr[1:])):
        order = np.argsort(ts_arr, kind="mergesort")
        ts_arr = ts_arr[order]
        px_arr = px_arr[order]

    prev_px = px_arr[:-2]
    cur_px = px_arr[1:-1]
    next_px = px_arr[2:]
    lo_nb = np.minimum(prev_px, next_px)
    hi_nb = np.maximum(prev_px, next_px)
    agree = (lo_nb > 0.0) & ((hi_nb / lo_nb) <= _WINDOW_OUTLIER_NEIGHBOR_AGREE_RATIO)
    spike = agree & (
        (cur_px > hi_nb * _WINDOW_OUTLIER_SPIKE_RATIO)
        | (cur_px < lo_nb / _WINDOW_OUTLIER_SPIKE_RATIO)
    )
    removed = np.count_nonzero(spike)
    if removed > 0:
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[1:-1] = ~spike
        kept_ts = ts_arr[keep_mask]
        kept_px = px_arr[keep_mask]
    else:
        kept_ts = ts_arr
        kept_px = px_arr

    if kept_px.size >= 5:
        med_px = np.median(kept_px).item()
        if med_px > 0.0:
            extreme_indices = np.flatnonzero(
                np.maximum(kept_px / med_px, med_px / kept_px) > _WINDOW_OUTLIER_MEDIAN_RATIO
            )
            if extreme_indices.size > 0:
                drop_ext = np.zeros(kept_px.size, dtype=bool)
                for idx_raw in extreme_indices:
                    idx = idx_raw.item()
                    px = kept_px[idx].item()
                    corr = (
                        np.count_nonzero(
                            np.maximum(kept_px / px, px / kept_px) <= _WINDOW_OUTLIER_CORROBORATE_RATIO
                        )
                        - 1
                    )
                    if corr < 2:
                        drop_ext[idx] = True
                ext_removed = np.count_nonzero(drop_ext)
                if ext_removed > 0:
                    kept_ts = kept_ts[~drop_ext]
                    kept_px = kept_px[~drop_ext]
                    removed += ext_removed
    return kept_ts, kept_px, removed


def _get_symbol_window_extrema(symbol, current_price,
                               minutes_back=ASSET_REFERENCE_MINUTES_BACK_DEFAULT):
    """Return validated per-symbol price minimum/maximum for the rolling window."""
    symbol = str(symbol or "").strip().upper()
    minutes_back = _finite_float(minutes_back, positive=True)
    current_price = _finite_float(current_price, positive=True)
    if minutes_back is None or current_price is None:
        return _log_symbol_and_return(symbol, "Invalid price/cache window; skip evaluation.", None)
    now_ts = time.time() * 1.0
    target_ts = now_ts - minutes_back * 60.0

    cached_ts, cached_px = _extract_window_arrays(_read_symbol_price_rows(symbol), target_ts, now_ts)
    cached_ts, cached_px = _sort_and_dedup_ts_px(cached_ts, cached_px)

    if cached_px.size >= 5:
        clean_ts, clean_px, _ = _filter_isolated_price_outliers_arrays(cached_ts, cached_px)
        if clean_px.size >= 5:
            hist_med = np.median(clean_px).item()
            recent_tail = clean_px[-5:]
            recent_matches = np.count_nonzero(
                np.maximum(recent_tail / current_price, current_price / recent_tail)
                <= _WINDOW_OUTLIER_CORROBORATE_RATIO
            )
            if (
                hist_med > 0.0
                and max(current_price / hist_med, hist_med / current_price)
                > _WINDOW_OUTLIER_MEDIAN_RATIO
                and recent_matches < 2
            ):
                return _log_symbol_and_return(
                    symbol,
                    f"Current price {current_price:.8f} rejected as outlier vs rolling median "
                    f"{hist_med:.8f}; skip evaluation.",
                    None,
                )

    local = _local_price_samples.setdefault(symbol, [])
    local.append([now_ts, current_price])
    local[:] = [row for row in local
                if (parsed := _price_row(row)) is not None
                and target_ts <= parsed[0] <= now_ts]

    local_ts, local_px = _extract_window_arrays(local, target_ts, now_ts)
    if cached_ts.size and local_ts.size:
        comb_ts = np.concatenate((cached_ts, local_ts))
        comb_px = np.concatenate((cached_px, local_px))
    elif local_ts.size:
        comb_ts, comb_px = local_ts, local_px
    else:
        comb_ts, comb_px = cached_ts, cached_px

    comb_ts, comb_px = _sort_and_dedup_ts_px(comb_ts, comb_px)
    norm_ts, norm_px, removed_outliers = _filter_isolated_price_outliers_arrays(comb_ts, comb_px)
    if removed_outliers > 0:
        print(
            f"[WARNING] [{symbol}] Filtered {removed_outliers} isolated price outlier(s) "
            f"from rolling window."
        )

    print(f"[DEBUG] {symbol} candidate price rows: {norm_px.size}")
    if norm_px.size < 2:
        return _log_symbol_and_return(symbol, "Insufficient per-asset price baseline; skip evaluation.", None)
    vol_mult = _compute_symbol_volatility_multiplier_arrays(norm_ts, norm_px)
    eval_row = _last_evaluation.setdefault(
        symbol,
        {"growth": None, "drawdown": None, "pending_tier": False, "volatility_mult": 1.0},
    )
    eval_row["volatility_mult"] = vol_mult
    idx_min = np.argmin(norm_px).item()
    idx_max = np.argmax(norm_px).item()
    minimum = {"timestamp": norm_ts[idx_min].item(), "price": norm_px[idx_min].item()}
    maximum = {"timestamp": norm_ts[idx_max].item(), "price": norm_px[idx_max].item()}
    print(f"[DEBUG] {symbol} chosen MIN/MAX: {minimum}/{maximum} (vol_mult={vol_mult:.4f})")
    return minimum, maximum


def _read_total_asset_value_rows():
    """Copy shared TOTAL portfolio USDC history from AssetValue cache."""
    try:
        manager = get_asset_value_manager(symbols=["TOTAL"])
        if manager is None:
            return []
        if hasattr(manager, "resync_mem_file"):
            manager.resync_mem_file()
        with manager.lock:
            rows = list(manager.cache.get("TOTAL", []))
        return rows
    except Exception as exc:
        print(f"ERROR reading AssetValue cache: {exc}")
        return []


def _get_portfolio_growth_percent(minutes_back=ASSET_REFERENCE_MINUTES_BACK_DEFAULT):
    """Compute percentage growth of TOTAL account value over the rolling window."""
    minutes_back = _finite_float(minutes_back, positive=True)
    if minutes_back is None:
        return None
    target_ts = (now_ts := time.time() * 1.0) - minutes_back * 60.0
    rows = _read_total_asset_value_rows()
    valid = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        payload = row.get("i") if isinstance(row.get("i"), dict) else row
        ts = extract_timestamp_s(payload, default=None)
        val = (
            _finite_float(payload.get("total_value_usdc"), positive=True)
            or _finite_float(payload.get("total_value_usdt"), positive=True)
        )
        if ts is None or val is None:
            continue
        if target_ts <= ts <= now_ts:
            valid.append((ts, val))
    if len(valid) < 2:
        return None
    min_val = min(v for _, v in valid)
    latest_val = max(valid, key=lambda item: item[0])[1]
    if min_val <= 0:
        return None
    return ((latest_val - min_val) / min_val) * 100.0


def _read_p3b_sentiment(path=None, now_ts=None):
    """Read fresh Pillar 3B SentimentAdvisor assessment if available and within TTL."""
    if path is None:
        state_path = getattr(STATE, "path", None)
        if not state_path:
            return None
        cache_dir = os.path.dirname(os.path.abspath(state_path))
        data = load_cached_sentiment_advisor_eval(cache_dir)
    else:
        data = read_json_dict(path, default=None, recover_previous=True)
    if not isinstance(data, dict):
        return None

    eval_ts = _finite_float(data.get("ts"), positive=True)
    now_val = _finite_float(now_ts, positive=True) or (time.time() * 1.0)
    if eval_ts is None:
        return None
    age_sec = now_val - eval_ts
    if age_sec < -60.0 or age_sec > P3B_MAX_AGE_SECONDS:
        return None

    return {
        "recommended_action": str(data.get("recommended_action") or "HOLD").strip().upper(),
        "market_bias": str(data.get("market_bias") or "NEUTRAL").strip().upper(),
        "risk_level": str(data.get("risk_level") or "MODERATE").strip().upper(),
        "confidence": clamp_float(data.get("confidence"), 0.0, 1.0, default=0.0),
        "ts": eval_ts,
    }


def _cycle_portfolio_growth_percent(minutes_back=ASSET_REFERENCE_MINUTES_BACK_DEFAULT):
    cycle = getattr(_CYCLE_SCOPE, "cache", None)
    if cycle is None:
        return _get_portfolio_growth_percent(minutes_back=minutes_back)
    cache_map = cycle["portfolio_growth"]
    if minutes_back not in cache_map:
        cache_map[minutes_back] = _get_portfolio_growth_percent(minutes_back=minutes_back)
    return cache_map[minutes_back]


def _cycle_p3b_sentiment():
    cycle = getattr(_CYCLE_SCOPE, "cache", None)
    if cycle is None:
        return _read_p3b_sentiment()
    if not cycle["p3b_loaded"]:
        cycle["p3b_val"] = _read_p3b_sentiment()
        cycle["p3b_loaded"] = True
    return copy.deepcopy(cycle["p3b_val"])


def _p3b_threshold_multipliers(assessment=None, bypass_guards=None):
    """Compute (sell_mult, buy_mult, assessment) from Pillar 3B sentiment state.

    Preserves canonical tier identities while scaling trigger thresholds:
    - SELL: DEFENSIVE compresses up to AG_P3B_SELL_DEFENSIVE_MAX_DISCOUNT (down to 50%),
      TRIM_PROFITS compresses up to AG_P3B_SELL_TRIM_MAX_DISCOUNT (down to 65%),
      weighted by LLM confidence. BULLISH/HOLD keeps 1.0x static thresholds.
    - BUY: ACCUMULATE (BULLISH) compresses drawdown thresholds by up to
      AG_P3B_BUY_ACCUMULATE_MAX_DISCOUNT; DEFENSIVE expands drawdown thresholds by
      up to AG_P3B_BUY_DEFENSIVE_MAX_PENALTY to avoid catching falling knives early.
    """
    eff_bypass = _og.resolve_effective_bypass_guards("binance", bypass_guards=bypass_guards)
    if GUARD_REGISTRY["GLLM2"].is_bypassed(eff_bypass):
        return 1.0, 1.0, None
    if assessment is None:
        assessment = _cycle_p3b_sentiment()
    if not isinstance(assessment, dict):
        return 1.0, 1.0, None

    action = str(assessment.get("recommended_action") or "HOLD").strip().upper()
    bias = str(assessment.get("market_bias") or "NEUTRAL").strip().upper()
    risk = str(assessment.get("risk_level") or "MODERATE").strip().upper()
    conf = clamp_float(assessment.get("confidence"), 0.0, 1.0, default=0.0)

    # 1. SELL threshold multiplier
    if action == "DEFENSIVE" or (bias == "BEARISH" and risk in {"HIGH", "EXTREME"}):
        sell_discount = P3B_SELL_DEFENSIVE_MAX_DISCOUNT * conf
    elif action == "TRIM_PROFITS" or (bias in {"BEARISH", "CAUTION"} and risk in {"MODERATE", "HIGH", "EXTREME"}):
        sell_discount = P3B_SELL_TRIM_MAX_DISCOUNT * conf
    else:
        sell_discount = 0.0
    sell_mult = max(1.0 - P3B_SELL_DEFENSIVE_MAX_DISCOUNT, 1.0 - sell_discount)

    # 2. BUY threshold multiplier
    if action == "ACCUMULATE" and bias == "BULLISH" and risk not in {"HIGH", "EXTREME"}:
        buy_mult = max(
            1.0 - P3B_BUY_ACCUMULATE_MAX_DISCOUNT,
            1.0 - P3B_BUY_ACCUMULATE_MAX_DISCOUNT * conf,
        )
    elif action == "DEFENSIVE" or (bias == "BEARISH" and risk in {"HIGH", "EXTREME"}):
        buy_mult = min(
            1.0 + P3B_BUY_DEFENSIVE_MAX_PENALTY,
            1.0 + P3B_BUY_DEFENSIVE_MAX_PENALTY * conf,
        )
    elif action == "TRIM_PROFITS" or bias in {"BEARISH", "CAUTION"}:
        buy_mult = min(
            1.0 + P3B_BUY_DEFENSIVE_MAX_PENALTY,
            1.0 + 0.5 * P3B_BUY_DEFENSIVE_MAX_PENALTY * conf,
        )
    else:
        buy_mult = 1.0

    return sell_mult, buy_mult, assessment


def _effective_portfolio_growth_threshold(sell_mult=1.0):
    """Return effective 24h portfolio growth threshold (%) scaled by P3B sell_mult."""
    return PORTFOLIO_GROWTH_PERCENT * safe_pos_float(sell_mult, default=1.0)


def _trigger_threshold_for_sell_tier(
    canonical_threshold, sell_mult=1.0, portfolio_growth_pct=None, volatility_mult=1.0
):
    """Return effective growth trigger (%) for a canonical SELL tier.

    - The hybrid 4.1% tier (already unlocked by portfolio growth) is compressed by
      P3B sell_mult down to a floor of SELL_REARM_GROWTH_PERCENT + 1.5 (3.5%).
    - Static SELL tiers (9, 12, 15, 25, 35%) are scaled by P3B sell_mult,
      portfolio growth multiplier, and 24h symbol volatility multiplier down to
      SELL_REARM_GROWTH_PERCENT + 2.0 (4.0%), preserving strict ordering above
      the hybrid tier.
    """
    default_sell_thr = safe_pos_float(SELL_TIERS[0][0], default=1.0) or 1.0
    thr = safe_pos_float(canonical_threshold, default=default_sell_thr) or default_sell_thr
    sm = safe_pos_float(sell_mult, default=1.0)
    vm = _clamp_volatility_mult(volatility_mult)
    static_thresholds = {
        tf for t, _ in SELL_TIERS if (tf := safe_pos_float(t, default=None)) is not None
    }
    if thr not in static_thresholds and abs(thr - PORTFOLIO_ASSET_MIN_GROWTH_PERCENT) <= 1e-9:
        return max(SELL_REARM_GROWTH_PERCENT + 1.5, thr * sm)
    pg = _finite_float(portfolio_growth_pct)
    pg_mult = (
        1.0 - PORTFOLIO_SELL_MAX_DISCOUNT
        if pg is not None and pg >= _effective_portfolio_growth_threshold(sm)
        else 1.0
    )
    return max(
        SELL_REARM_GROWTH_PERCENT + 2.0,
        thr * sm * (pg_mult * 1.0) * vm,
    )


def _trigger_threshold_for_buy_tier(
    canonical_threshold, buy_mult=1.0, volatility_mult=1.0
):
    """Return effective drawdown trigger (%) for a canonical BUY tier."""
    default_buy_thr = safe_pos_float(BUY_TIERS[0][0], default=1.0) or 1.0
    thr = safe_pos_float(canonical_threshold, default=default_buy_thr) or default_buy_thr
    bm = safe_pos_float(buy_mult, default=1.0)
    vm = _clamp_volatility_mult(volatility_mult)
    return max(RECOVERY_RESET_PERCENT + 0.5, thr * bm * vm)


_LAST_SUBMIT_OUTCOME: dict = {}


def _refuse_submit(reason: str, ret_val=None, message: str | None = None):
    if message is not None:
        print(message)
    _LAST_SUBMIT_OUTCOME.update(accepted=False, state="refused", reason=reason)
    return ret_val


def sell_asset(symbol, qty, current_price, client_order_id, tier_threshold):
    """Submit one explicit SELL tier and return the native order response."""
    _LAST_SUBMIT_OUTCOME.clear()
    try:
        base_asset, _ = resolve_assets(symbol)
        provider = mkt.provider_by_name("binance")
        if provider is None:
            return _refuse_submit(
                "provider_unavailable", None, f"[{symbol}] Binance provider unavailable; skip SELL fail-closed."
            )
        free_qty = _finite_float(provider.free_balance(base_asset), positive=True)
        requested_qty = _finite_float(qty, positive=True)
        price = _finite_float(current_price, positive=True)
    except Exception as e:
        return _refuse_submit("prepare_error", None, f"ERROR preparing SELL for {symbol}: {e}")
    if free_qty is None:
        return _refuse_submit("no_free_balance", None, f"[{symbol}] No valid free {base_asset} balance; skip SELL.")
    if requested_qty is None or price is None:
        return _refuse_submit("invalid_price_or_qty", None, f"[{symbol}] Invalid SELL price/quantity; skip SELL.")
    qty_to_sell = min(requested_qty, free_qty)
    if qty_to_sell <= 0:
        return _refuse_submit("zero_qty", None, f"[{symbol}] SELL tier has no reconciled free quantity.")
    try:
        order = mkt.place(
            symbol, "SELL", price, qty_to_sell,
            force=False, smart=False, caller_owns_retry=True,
            bypass_guards={"G_QLIMIT1"}, module="assetguardian",
            wait_for_trend=True, client_order_id=client_order_id,
            motivation=f"assetguardian_growth_exit_tier_{tier_threshold:g}",
            _outcome_context=_LAST_SUBMIT_OUTCOME,
        )
        if order:
            print(
                f" SELL tier submitted: {symbol} requested={requested_qty:.8f} "
                f"free={free_qty:.8f} submitted_cap={qty_to_sell:.8f} clientOrderId={client_order_id}"
            )
            return order
        print(f" SELL tier not accepted: {symbol} qty={qty_to_sell}")
        return None
    except Exception as e:
        print(f"ERROR selling {symbol}: {e}")
        return None


def buy_with_all_cash(
    buy_symbol, cash_ratio=BUY_USE_CASH_RATIO, cash_amount=None, client_order_id=None, respect_guard=False
):
    _LAST_SUBMIT_OUTCOME.clear()
    try:
        _, quote_asset = resolve_assets(buy_symbol)
        cash_ratio = _finite_float(cash_ratio, positive=True)
        if cash_ratio is None or cash_ratio > 1:
            return _refuse_submit("invalid_cash_ratio", False, f" Invalid cash ratio for buy: {cash_ratio}")
        provider = mkt.provider_by_name("binance")
        if provider is None:
            return _refuse_submit(
                "provider_unavailable", False, f"[{buy_symbol}] Binance provider unavailable; skip BUY fail-closed."
            )
        free_cash = _finite_float(provider.free_balance(quote_asset), positive=True)
        current_price = _finite_float(api.get_current_price(buy_symbol), positive=True)
    except Exception as e:
        return _refuse_submit("prepare_error", False, f"ERROR preparing buy for {buy_symbol}: {e}")
    print(f"[DEBUG] buy check symbol={buy_symbol}, quote={quote_asset}, free_cash={free_cash}, current_price={current_price}")

    if free_cash is None:
        return _refuse_submit("no_free_cash", False, f" No valid available {quote_asset} balance; skip buy fail-closed.")
    if current_price is None:
        return _refuse_submit("invalid_price", False, f" Invalid current price for {buy_symbol}.")

    requested_cash = free_cash * cash_ratio if cash_amount is None else _finite_float(cash_amount, positive=True)
    if requested_cash is None:
        return _refuse_submit("invalid_requested_cash", False, " Invalid requested cash amount for buy.")
    cash_to_use = min(requested_cash, free_cash * BUY_USE_CASH_RATIO)
    qty = cash_to_use / current_price
    if qty <= 0:
        return _refuse_submit("zero_qty", False, " Computed qty <= 0. Skip buy.")

    print(
        f" BUY trigger active -> symbol={buy_symbol}, using {cash_to_use:.6f} {quote_asset} "
        f"(safe cap {cash_ratio*100:.2f}% of free cash), qty={qty:.8f}"
    )
    try:
        bypass_guards = (
            {"G_QLIMIT1"}
            if _parse_bool_flag(respect_guard, default=False)
            else {"G_PROF", "G_QLIMIT1", "G_QLIMIT2", "GI_OBOOK", "GI_WHALE", "GI_FUNDING", "GLLM1", "GLLM2", "GLLM3"}
        )
        order = mkt.place(
            buy_symbol, "BUY", current_price, qty,
            force=False, smart=False, caller_owns_retry=True,
            bypass_guards=bypass_guards, module="assetguardian",
            wait_for_trend=True, client_order_id=client_order_id,
            motivation="assetguardian_drawdown_buy",
            _outcome_context=_LAST_SUBMIT_OUTCOME,
        )
        if order:
            print(f" BUY safe-order sent: {buy_symbol}, qty={qty:.8f}")
            return order
        print(f" BUY safe-order failed: {buy_symbol}, qty={qty:.8f}")
        return None
    except Exception as e:
        print(f"ERROR buying {buy_symbol}: {e}")
        return None


def _completed_tier_values(state):
    completed = set()
    if not isinstance(state, dict):
        return completed
    raw_tiers = state.get("completed_tiers", [])
    if not isinstance(raw_tiers, (list, tuple, set)):
        return completed
    for raw in raw_tiers:
        value = _finite_float(raw, positive=True)
        if value is not None:
            completed.add(value)
    return completed


def _load_state_root():
    """Load v3 BUY/SELL state and conservatively migrate older BUY-only data."""
    cycle = getattr(_CYCLE_SCOPE, "cache", None)
    if cycle is not None and cycle["state_id"] == id(STATE) and isinstance(cycle["state_root"], dict):
        return copy.deepcopy(cycle["state_root"])
    raw = STATE.load()
    if not isinstance(raw, dict):
        raw = {}
    symbol_rows = raw.get("symbols")
    if isinstance(symbol_rows, dict):
        normalized = {}
        for symbol, value in symbol_rows.items():
            if not isinstance(value, dict):
                continue
            if "buy" in value or "sell" in value:
                buy = value.get("buy")
                sell = value.get("sell")
                row = {}
                if isinstance(buy, dict) and buy:
                    row["buy"] = dict(buy)
                if isinstance(sell, dict) and sell:
                    row["sell"] = dict(sell)
            else:
                # Version 2 stored the BUY campaign directly under the symbol.
                row = {"buy": dict(value)} if value else {}
            if row:
                normalized[str(symbol)] = row
        root = {"version": 3, "symbols": normalized}
    elif {"peak_value", "peak_ts", "initial_cash", "completed_tiers"}.intersection(raw):
        root = {"version": 3, "symbols": {LEGACY_BUY_SYMBOL: {"buy": dict(raw)}}}
    else:
        root = {"version": 3, "symbols": {}}
    if cycle is not None:
        cycle["state_id"] = id(STATE)
        cycle["state_root"] = copy.deepcopy(root)
    return root


def _symbol_campaign(symbol, kind):
    row = casefold_get(_load_state_root()["symbols"], symbol, {})
    campaign = row.get(kind, {}) if isinstance(row, dict) else {}
    return dict(campaign) if isinstance(campaign, dict) else {}


def _has_pending_campaign(campaign: object) -> bool:
    return isinstance(campaign, dict) and isinstance(campaign.get("pending"), dict) and bool(campaign["pending"])


def _save_symbol_campaign(symbol, kind, campaign):
    if kind not in {"buy", "sell"}:
        raise ValueError(f"unknown AssetGuardian campaign: {kind}")
    root = _load_state_root()
    symbols_state = dict(root["symbols"])
    sym_cf = str(symbol or "").strip().casefold()
    row = {}
    for k in list(symbols_state):
        if isinstance(k, str) and k.strip().casefold() == sym_cf:
            existing = symbols_state.pop(k, None)
            if isinstance(existing, dict):
                row.update(existing)
    if campaign:
        row[kind] = dict(campaign)
    else:
        row.pop(kind, None)
    if row:
        symbols_state[symbol] = row
    new_root = {"version": 3, "symbols": symbols_state}
    cycle = getattr(_CYCLE_SCOPE, "cache", None)
    try:
        STATE.save(new_root)
    except Exception:
        if cycle is not None:
            cycle["state_id"], cycle["state_root"] = None, None
        raise
    if cycle is not None:
        cycle["state_id"] = id(STATE)
        cycle["state_root"] = copy.deepcopy(new_root)


def _clear_campaign_trend_defer(symbol, kind, state):
    if not isinstance(state, dict) or "trend_defer" not in state:
        return state
    cleaned = dict(state)
    cleaned.pop("trend_defer", None)
    _save_symbol_campaign(symbol, kind, cleaned)
    return cleaned


def _trend_defer_ready(symbol, kind, side, threshold, state):
    """Non-blocking trend deferral, re-evaluated by the normal Guardian loop."""
    state = dict(state)
    now = time.time() * 1.0
    thr_f = require_pos_float(threshold, "threshold")
    defer = state.get("trend_defer")
    if not isinstance(defer, dict):
        defer = {}
    same_intent = (
        str(defer.get("side") or "").upper() == str(side).upper()
        and _finite_float(defer.get("threshold"), positive=True) == thr_f
    )
    started_at = (_finite_float(defer.get("started_at"), positive=True)
                  if same_intent else None)
    try:
        short_tm = cm.get_short_trend_manager()
        should_wait = pw.should_wait_short_trend(
            short_tm, side, symbol, short_trend_cls=cm.CacheShortTrendManager
        )
    except Exception as exc:
        print(f"[{symbol}] Trend defer unavailable; continue submit: {exc}")
        should_wait = False

    if should_wait:
        if started_at is None or started_at > now:
            started_at = now
            state["trend_defer"] = {
                "side": str(side).upper(),
                "threshold": thr_f,
                "started_at": started_at,
            }
            _save_symbol_campaign(symbol, kind, state)
        elapsed = max(0.0, now - started_at)
        print(
            f"[{symbol}] {side} tier {thr_f:g}% deferred by "
            f"short trend (GT_S): {elapsed:.1f}s elapsed; "
            "waiting for trend reversal without blocking sleep.")
        return False, state

    return True, _clear_campaign_trend_defer(symbol, kind, state)


def _reconcile_campaign_initial_amount(
    symbol: str,
    kind: str,
    state: dict,
    free_amount: float,
    *,
    open_sells_loader=None,
) -> dict:
    """Reconcile ``initial_cash`` (BUY) or ``initial_qty`` (SELL) against free and locked balance."""
    init_key, used_key, min_tol, label = (
        ("initial_cash", "total_spent_quote", 1e-9, "Free quote balance")
        if kind == "buy"
        else ("initial_qty", "total_filled_qty", 1e-12, "Free balance")
    )
    init_val = _finite_float(state.get(init_key), positive=True)
    if init_val is None:
        return state
    used_val = safe_nonneg_float(state.get(used_key), default=0.0)
    expected_rem = max(0.0, init_val - used_val)
    tol = max(min_tol, init_val * 1e-6)
    if free_amount > expected_rem + tol:
        reconciled = free_amount + used_val
        print(
            f"[{symbol}] {label} increased from expected {expected_rem:.8f} to {free_amount:.8f}; "
            f"reconciling {kind.upper()} campaign {init_key} from {init_val:.8f} "
            f"to {reconciled:.8f} while preserving stage tracking."
        )
        state = {**state, init_key: reconciled}
        _save_symbol_campaign(symbol, kind, state)
    elif (
        kind == "sell"
        and callable(open_sells_loader)
        and (not _completed_tier_values(state) or used_val > 0.0)
        and free_amount + tol < expected_rem
    ):
        open_sells = open_sells_loader()
        if open_sells is not None and all(
            extract_remaining_qty(o, default=0.0) > 0.0 for o in open_sells
        ):
            locked_qty = sum(extract_remaining_qty(o, default=0.0) for o in open_sells)
            total_held = free_amount + locked_qty
            if total_held + tol < expected_rem:
                reconciled = total_held + used_val
                print(
                    f"[{symbol}] Total held base ({total_held:.8f} = free {free_amount:.8f} + "
                    f"open_sell {locked_qty:.8f}) decreased below expected {expected_rem:.8f}; "
                    f"reconciling SELL campaign initial_qty from {init_val:.8f} "
                    f"to {reconciled:.8f} while preserving stage tracking."
                )
                state = {**state, init_key: reconciled}
                _save_symbol_campaign(symbol, kind, state)
    return state


def _campaign_tier(symbol, drawdown_abs, maximum_row, free_cash,
                   p3b_assessment=None):
    """Select one crossed, incomplete BUY tier for one symbol campaign."""
    state = _symbol_campaign(symbol, "buy")
    if drawdown_abs < RECOVERY_RESET_PERCENT:
        if state:
            _save_symbol_campaign(symbol, "buy", {})
        return None, {}

    peak_price = _finite_float(maximum_row.get("price"), positive=True)
    peak_ts = _finite_float(maximum_row.get("timestamp"), positive=True)
    stored_peak = _finite_float(state.get("peak_price"), positive=True)
    if (peak_price is None or peak_ts is None
            or _finite_float(free_cash, positive=True) is None):
        return None, state
    if (
        state
        and stored_peak is not None
        and not _has_pending_campaign(state)
        and _is_corrupted_buy_peak(stored_peak, state.get("peak_ts"), peak_price)
    ):
        print(
            f"[WARNING] [{symbol}] Resetting corrupted BUY campaign peak_price={stored_peak:.8f} "
            f"(validated window max={peak_price:.8f})."
        )
        _save_symbol_campaign(symbol, "buy", {})
        state = {}
        stored_peak = None
    if not state:
        state = {
            "peak_price": peak_price,
            "peak_ts": peak_ts,
            "initial_cash": free_cash,
            "completed_tiers": [],
            "spent_quote_by_tier": {},
            "total_spent_quote": 0.0,
            "attempts_by_tier": {},
            "terminal_orders": [],
        }
        _save_symbol_campaign(symbol, "buy", state)
    elif stored_peak is None or peak_price > stored_peak:
        # Legacy global campaign migration (stored_peak is None) or a higher peak
        # within the same unrecovered campaign: update peak reference without
        # clearing completed tiers or replenishing budget.
        state = {**state, "peak_price": peak_price, "peak_ts": peak_ts}
        if stored_peak is None:
            if _finite_float(state.get("initial_cash"), positive=True) is None:
                state["initial_cash"] = free_cash
            state["completed_tiers"] = sorted(_completed_tier_values(state))
        _save_symbol_campaign(symbol, "buy", state)
    if stored_peak is not None:
        state = _reconcile_campaign_initial_amount(symbol, "buy", state, free_cash)

    # Try the deepest crossed tier first. The shallowest tier respects the profit
    # guard and can be refused on every cycle; walking tiers from the top made that
    # refused tier the permanent pick, so the deeper tiers designed to bypass the
    # guard were never attempted. Per-tier budgets are unchanged by the order.
    _, buy_mult, _ = _p3b_threshold_multipliers(p3b_assessment)
    vol_mult = _symbol_volatility_mult(symbol)
    return _first_due_buy_tier(drawdown_abs, _completed_tier_values(state), buy_mult, vol_mult), state


def _first_due_buy_tier(drawdown_abs: float | None, completed: set, buy_mult: float = 1.0, vol_mult: float = 1.0):
    if drawdown_abs is None:
        return None
    return next(
        (
            (threshold, allocation)
            for threshold, allocation in sorted(BUY_TIERS, key=lambda tier: tier[0], reverse=True)
            if threshold not in completed
            and drawdown_abs >= _trigger_threshold_for_buy_tier(threshold, buy_mult, volatility_mult=vol_mult)
        ),
        None,
    )


def _first_due_sell_tier(
    growth: float | None,
    completed: set,
    sell_mult: float = 1.0,
    portfolio_growth_pct: float | None = None,
    vol_mult: float = 1.0,
    active_tiers=None,
):
    if growth is None:
        return None
    tiers = active_tiers if active_tiers is not None else _effective_sell_tiers(portfolio_growth_pct, sell_mult=sell_mult)
    return next(
        (
            (threshold, allocation)
            for threshold, allocation in tiers
            if threshold not in completed
            and growth >= _trigger_threshold_for_sell_tier(
                threshold, sell_mult, portfolio_growth_pct, volatility_mult=vol_mult
            )
        ),
        None,
    )


def _complete_campaign_tier(symbol, kind, state, threshold):
    thr_f = _finite_float(threshold, positive=True)
    if thr_f is None:
        return
    completed = _completed_tier_values(state)
    completed.add(thr_f)
    state = dict(state)
    state["completed_tiers"] = sorted(completed)
    _save_symbol_campaign(symbol, kind, state)


def _tier_key(threshold):
    return f"{require_pos_float(threshold, 'threshold'):g}"


def _buy_remaining_cash(state, threshold, allocation):
    initial_cash = _finite_float(state.get("initial_cash"), positive=True)
    if initial_cash is None:
        return None, None, None
    alloc_f = require_pos_float(allocation, "allocation")
    target_cash = initial_cash * BUY_USE_CASH_RATIO * alloc_f
    raw_spent_map = state.get("spent_quote_by_tier")
    spent_map = dict(raw_spent_map) if isinstance(raw_spent_map, dict) else {}
    prior_spent = max(0.0, _finite_float(spent_map.get(_tier_key(threshold))) or 0.0)
    return max(0.0, target_cash - prior_spent), target_cash, prior_spent


def _ag_client_order_id(kind, symbol, state, threshold, attempt):
    prefix, anchor_key = ("AGS", "trough_ts") if kind == "sell" else ("AGB", "peak_ts")
    thr_f = require_pos_float(threshold, "threshold")
    att_i = require_int(attempt, "attempt", min_value=0)
    raw = f"assetguardian:{kind}:{symbol}:{state.get(anchor_key)}:{thr_f:g}:{att_i}"
    return prefix + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:29]


def _finalize_terminal_campaign_state(
    symbol,
    kind,
    state,
    pending,
    threshold,
    status,
    filled_qty,
    cost_val,
    fee_val,
    *,
    dust_completed=False,
    target_amount: float | None = None,
    tier_accum: float = 0.0,
    min_tol: float = 1e-9,
):
    is_dust_done = bool(dust_completed) or (
        target_amount is not None
        and tier_accum > 0
        and (target_amount - tier_accum) <= max(min_tol, target_amount * 0.02)
    )
    if (status.status == "closed" and filled_qty > 0) or is_dust_done:
        completed = _completed_tier_values(state)
        completed.add(threshold)
        state["completed_tiers"] = sorted(completed)
    raw_terminal = state.get("terminal_orders")
    terminal = list(raw_terminal)[-19:] if isinstance(raw_terminal, (list, tuple)) else []
    terminal.append({
        "order_id": pending.get("order_id"),
        "client_order_id": pending.get("client_order_id"),
        "threshold": threshold,
        "status": status.status,
        "filled_qty": filled_qty,
        "cost": cost_val,
        "fee": fee_val,
        "terminal_ts": time.time(),
    })
    state["terminal_orders"] = terminal
    state.pop("pending", None)
    _save_symbol_campaign(symbol, kind, state)
    print(
        f"[{symbol}] {kind.upper()} tier terminal: threshold={threshold:g}% status={status.status} "
        f"orderId={pending.get('order_id')} filled={filled_qty:.8f} cost={cost_val:.8f} fee={fee_val:.8f}"
    )
    return state


def _extract_and_accumulate_terminal_tier(
    symbol,
    state: dict,
    status,
    kind_label: str,
    map_key: str,
    total_key: str,
    *,
    include_cost_and_fee: bool = False,
) -> tuple[dict, dict, float, float, float, float, float] | None:
    pending = dict(state.get("pending") or {})
    threshold = _finite_float(pending.get("threshold"), positive=True)
    if threshold is None:
        print(f"[{symbol}] Corrupt {kind_label}; preserve fail-closed.")
        return None
    key = _tier_key(threshold)
    filled_qty = safe_nonneg_float(status.filled_qty, default=0.0)
    cost_val = safe_nonneg_float(status.cost, default=0.0)
    fee_val = safe_nonneg_float(status.fee, default=0.0)
    delta = max(0.0, cost_val + fee_val) if include_cost_and_fee else filled_qty
    raw_map = state.get(map_key)
    by_tier = dict(raw_map) if isinstance(raw_map, dict) else {}
    tier_accum = safe_nonneg_float(by_tier.get(key), default=0.0) + delta
    by_tier[key] = tier_accum
    out = dict(state)
    out[map_key] = by_tier
    out[total_key] = safe_nonneg_float(state.get(total_key), default=0.0) + delta
    return out, pending, threshold, tier_accum, filled_qty, cost_val, fee_val


def _record_terminal_buy(symbol, state, status):
    """Atomically apply one terminal BUY result and acknowledge generic pending."""
    accum = _extract_and_accumulate_terminal_tier(
        symbol,
        state,
        status,
        "BUY terminal pending",
        "spent_quote_by_tier",
        "total_spent_quote",
        include_cost_and_fee=True,
    )
    if accum is None:
        return state
    state, pending, threshold, tier_spent, filled_qty, cost_val, fee_val = accum
    allocation = _finite_float(pending.get("allocation"), positive=True)
    initial_cash = _finite_float(state.get("initial_cash"), positive=True)
    target_cash = (
        initial_cash * BUY_USE_CASH_RATIO * allocation
        if initial_cash is not None and allocation is not None
        else None
    )
    return _finalize_terminal_campaign_state(
        symbol,
        "buy",
        state,
        pending,
        threshold,
        status,
        filled_qty,
        cost_val,
        fee_val,
        target_amount=target_cash,
        tier_accum=tier_spent,
        min_tol=1e-9,
    )


def _record_terminal_sell(symbol, state, status):
    """Apply one terminal venue status without inferring acceptance as a fill."""
    accum = _extract_and_accumulate_terminal_tier(
        symbol,
        state,
        status,
        "SELL pending tier",
        "filled_qty_by_tier",
        "total_filled_qty",
    )
    if accum is None:
        return state
    state, pending, threshold, tier_filled, filled_qty, cost_val, fee_val = accum
    allocation = _finite_float(pending.get("allocation"), positive=True)
    if allocation is None:
        thr_f = safe_pos_float(threshold, default=0.0) or 0.0
        allocation = next(
            (
                alloc_v
                for t_thr, t_alloc in (*SELL_TIERS, (PORTFOLIO_ASSET_MIN_GROWTH_PERCENT, PORTFOLIO_SELL_ALLOCATION))
                if (t_thr_f := safe_pos_float(t_thr, default=None)) is not None
                and (alloc_v := safe_pos_float(t_alloc, default=None)) is not None
                and abs(t_thr_f - thr_f) <= 1e-9
            ),
            None,
        )
    initial_qty = _finite_float(state.get("initial_qty"), positive=True)
    target_qty = initial_qty * allocation if initial_qty is not None and allocation is not None else None
    return _finalize_terminal_campaign_state(
        symbol,
        "sell",
        state,
        pending,
        threshold,
        status,
        filled_qty,
        cost_val,
        fee_val,
        target_amount=target_qty,
        tier_accum=tier_filled,
        min_tol=1e-12,
    )


def _campaign_pending_persistor(symbol, kind, state):
    """Embed generic tracked pending state in one atomic campaign save."""
    if kind not in {"buy", "sell"}:
        raise ValueError(f"invalid tracked campaign kind: {kind}")
    box = {"state": dict(state)}

    def persist(pending):
        updated = dict(box["state"])
        if pending is None:
            updated.pop("pending", None)
        else:
            updated["pending"] = dict(pending)
        box["state"] = updated
        _save_symbol_campaign(symbol, kind, updated)

    return box, persist


def _reuse_unaccepted_attempt(symbol, kind, state, pending):
    """Release one unverified submit without consuming its deterministic ID.

    The next strategy evaluation increments the counter back to the same attempt,
    producing the same client_order_id.  This gives AssetGuardian at-least-once
    submit semantics while retaining venue-side idempotency where supported.
    """
    threshold = _finite_float(pending.get("threshold"), positive=True)
    attempt = safe_pos_int(pending.get("attempt"), default=0) or 0
    if threshold is None or attempt <= 0:
        print(f"[{symbol}] Cannot reuse corrupt {kind.upper()} attempt; preserve counters.")
        return state
    raw_attempts = state.get("attempts_by_tier")
    attempts = dict(raw_attempts) if isinstance(raw_attempts, dict) else {}
    attempts[_tier_key(threshold)] = max(0, attempt - 1)
    state = {**state, "attempts_by_tier": attempts}
    refusal_reason = pending.get("refusal_reason")
    if isinstance(refusal_reason, str) and refusal_reason.strip() and pending.get("intent_id"):
        state["last_refusal"] = {
            "intent_id": str(pending["intent_id"]),
            "reason": refusal_reason.strip(),
        }
    _save_symbol_campaign(symbol, kind, state)
    return state


def _reconcile_pending_campaign(symbol, kind, state):
    """Adapt generic tracked-order truth to BUY/SELL campaign accounting."""
    raw_pending = state.get("pending")
    if not isinstance(raw_pending, dict) or not raw_pending:
        return state, "none"
    side_label = kind.upper()
    record_terminal_fn = _record_terminal_sell if kind == "sell" else _record_terminal_buy
    pending = dict(raw_pending)
    pending.setdefault("intent_id", pending.get("client_order_id"))
    pending.setdefault("symbol", symbol)
    pending.setdefault("side", side_label)
    pending.setdefault("kind", f"ASSET_GUARDIAN_{side_label}_TIER")
    pending.setdefault("requested_price", None)
    box, persist = _campaign_pending_persistor(symbol, kind, state)
    if pending != raw_pending:
        persist(pending)
    try:
        result = ORDER_LIFECYCLE.reconcile(pending, persist=persist)
    except Exception as exc:
        print(f"[{symbol}] Invalid/corrupt {side_label} pending; block fail-closed: {exc}")
        return box["state"], "active"
    state = box["state"]
    if result.outcome == "terminal":
        return record_terminal_fn(symbol, state, result.status), "terminal"
    if result.outcome in {"absent", "retryable"}:
        state = _reuse_unaccepted_attempt(symbol, kind, state, result.intent)
        truth = ("confirmed absent" if result.outcome == "absent"
                 else "lookup unavailable")
        print(
            f"[{symbol}] {side_label} intent {truth}; release for same-ID, revalidated retry "
            "without completing tier.")
        return state, "retryable"
    status_label = result.status.status if result.status is not None else "unknown"
    print(
        f"[{symbol}] {side_label} tier pending: status={status_label} "
        f"orderId={result.intent.get('order_id')} "
        f"filled={_finite_float(result.intent.get('filled_qty')) or 0.0:.8f}")
    return state, "active"


def _active_sell_orders(provider, symbol):
    """Return active SELL orders, or ``None`` when venue truth is unavailable."""
    if provider is None:
        return None
    try:
        orders = provider.open_orders(symbol) or []
    except Exception as exc:
        print(f"[{symbol}] Open-order reconciliation unavailable: {exc}")
        return None
    return [
        order for order in orders
        if isinstance(order, dict)
        and extract_order_side(order) == "SELL"
    ]


_rtrade_sell_block_since: dict = {}


def _restore_preempted_orders(symbol: str, orders: list[dict], reason: str) -> int:
    if not orders:
        return 0

    class _MktProxy:
        @staticmethod
        def place(**kw):
            return mkt.place(symbol, kw.pop("side"), kw.pop("price"), kw.pop("qty"), **kw)

    return restore_cancelled_instrument_orders(
        _MktProxy,
        orders,
        bypass_guards={"all"},
        reason=reason,
    )


def _resolve_blocking_sell_orders(symbol, open_sells, *, canceled_orders_out: list | None = None):
    """Allow short retry when blocked only by lower-priority bots (rtrade, tradeall), then preemptively cancel."""
    if not open_sells:
        _rtrade_sell_block_since.pop(symbol, None)
        return []
    if any(not (isinstance(o, dict) and can_bot_preempt_order("assetguardian", o)) for o in open_sells):
        _rtrade_sell_block_since.pop(symbol, None)
        return open_sells
    now = time.time() * 1.0
    blocked_since = _rtrade_sell_block_since.get(symbol)
    if blocked_since is None or blocked_since > now:
        _rtrade_sell_block_since[symbol] = now
        print(
            f"[{symbol}] AssetGuardian SELL temporarily blocked by rtrade order(s) "
            f"{[extract_order_id(o) for o in open_sells]}; retrying before preemptive cancel "
            f"(0.0/{RTRADE_BLOCK_CANCEL_SECONDS:.1f}s)."
        )
        return open_sells
    elapsed = max(0.0, now - blocked_since)
    if elapsed < RTRADE_BLOCK_CANCEL_SECONDS:
        print(
            f"[{symbol}] AssetGuardian SELL still blocked by rtrade order(s) "
            f"{[extract_order_id(o) for o in open_sells]} "
            f"({elapsed:.1f}/{RTRADE_BLOCK_CANCEL_SECONDS:.1f}s); waiting before cancel."
        )
        return open_sells
    remaining = []
    for order in open_sells:
        oid = extract_order_id(order, default=None)
        if oid:
            try:
                if mkt.cancel_order(symbol, oid, provider_name="binance") is not False:
                    if canceled_orders_out is not None and isinstance(order, dict):
                        canceled_orders_out.append(dict(order))
                    print(
                        f"[{symbol}] Preemptively canceled blocking rtrade SELL order {oid} "
                        f"(cid={extract_client_order_id(order, default='')}) after {elapsed:.1f}s."
                    )
                    try:
                        resolve_accepted_order(symbol, oid)
                    except Exception:
                        pass
                    continue
                print(f"[{symbol}] Preemptive cancel of blocking rtrade SELL order {oid} returned False.")
            except Exception as exc:
                print(f"[{symbol}] Failed to cancel blocking rtrade SELL order {oid}: {exc}")
        remaining.append(order)
    if not remaining:
        _rtrade_sell_block_since.pop(symbol, None)
    return remaining


def _effective_sell_tiers(portfolio_growth_pct=None, sell_mult=1.0):
    """Include the hybrid portfolio+price_monitor tranche when total account value is up."""
    tiers = list(SELL_TIERS)
    pg = _finite_float(portfolio_growth_pct)
    if pg is not None and pg >= _effective_portfolio_growth_threshold(sell_mult):
        hybrid_thr = safe_pos_float(PORTFOLIO_ASSET_MIN_GROWTH_PERCENT, default=0.0) or 0.0
        if all(abs(t - hybrid_thr) > 1e-9 for t, _ in tiers):
            tiers.append((hybrid_thr, safe_pos_float(PORTFOLIO_SELL_ALLOCATION, default=0.0) or 0.0))
            tiers.sort()
    return tuple(tiers)


def _is_short_trend_waiting(symbol: str, side: str) -> bool:
    try:
        return bool(
            pw.should_wait_short_trend(
                cm.get_short_trend_manager(), side, symbol, short_trend_cls=cm.CacheShortTrendManager
            )
        )
    except Exception:
        return False


def _sell_campaign_tier(symbol, current_price, minimum_row, free_qty, provider,
                        portfolio_growth_pct=None, p3b_assessment=None,
                        canceled_orders_out: list | None = None):
    """Select one incomplete SELL tier against a frozen campaign trough."""
    state = _symbol_campaign(symbol, "sell")
    new_campaign = not bool(state)
    current_price = _finite_float(current_price, positive=True)
    free_qty = _finite_float(free_qty)
    if current_price is None or free_qty is None or free_qty < 0:
        return None, state, None
    sell_mult, _, _ = _p3b_threshold_multipliers(p3b_assessment)
    vol_mult = _symbol_volatility_mult(symbol)
    active_tiers = _effective_sell_tiers(portfolio_growth_pct, sell_mult=sell_mult)
    min_trigger_thr = min(
        _trigger_threshold_for_sell_tier(
            t, sell_mult, portfolio_growth_pct, volatility_mult=vol_mult
        )
        for t, _ in active_tiers
    )

    completed_existing = _completed_tier_values(state)

    def _has_due_sell_tier_at(g_val: float | None) -> bool:
        return (
            _first_due_sell_tier(
                g_val, completed_existing, sell_mult, portfolio_growth_pct, vol_mult, active_tiers=active_tiers
            )
            is not None
        )

    def _can_preempt_sell() -> bool:
        return canceled_orders_out is None or not _is_short_trend_waiting(symbol, "SELL")

    def _refresh_free_base() -> float | None:
        if not callable(getattr(provider, "free_balance", None)):
            return None
        try:
            base_asset, _ = resolve_assets(symbol)
            return _finite_float(provider.free_balance(base_asset), positive=True)
        except Exception:
            return None

    def _filter_blocking_sells(orders, *, needs_locked: bool = True) -> list[dict]:
        return [
            o for o in (orders or ())
            if resolve_priority_bot_owner(o) in {"assetguardian", "manual", "unspecified"}
            or (needs_locked and can_bot_preempt_order("assetguardian", o))
        ]

    preloaded_open_sells = None

    def _get_open_sells():
        nonlocal preloaded_open_sells
        if preloaded_open_sells is None:
            preloaded_open_sells = _active_sell_orders(provider, symbol)
        return preloaded_open_sells

    if free_qty == 0 or (free_qty * current_price) < 5.0:
        open_sells = _get_open_sells()
        if open_sells is None or open_sells:
            ref_trough = _finite_float((state or {}).get("trough_price") or minimum_row.get("price"), positive=True)
            cand_growth = (((current_price - ref_trough) / ref_trough) * 100.0) if ref_trough else None
            blocking_sells = _filter_blocking_sells(open_sells, needs_locked=True)
            if blocking_sells and _has_due_sell_tier_at(cand_growth) and _can_preempt_sell():
                rem_blocking = _resolve_blocking_sell_orders(
                    symbol, blocking_sells, canceled_orders_out=canceled_orders_out
                )
                if not rem_blocking:
                    canceled_refs = {id(o) for o in blocking_sells}
                    preloaded_open_sells = [o for o in (open_sells or []) if id(o) not in canceled_refs]
                    refreshed_free = _refresh_free_base()
                    if refreshed_free is not None and (refreshed_free * current_price) >= 5.0:
                        free_qty = refreshed_free
            if free_qty == 0 or (free_qty * current_price) < 5.0:
                print(f"[{symbol}] Zero free qty with unknown/active SELL orders; preserve state.")
                return None, state, None
        else:
            if state:
                _save_symbol_campaign(symbol, "sell", {})
            return None, {}, None

    def _reset_sell_campaign(growth_val=None):
        _save_symbol_campaign(symbol, "sell", {})
        return None, {}, growth_val

    if state:
        trough_price = _finite_float(state.get("trough_price"), positive=True)
        trough_ts = _finite_float(state.get("trough_ts"), positive=True)
        initial_qty = _finite_float(state.get("initial_qty"), positive=True)
        if trough_price is None or trough_ts is None or initial_qty is None:
            print(f"[{symbol}] Invalid SELL campaign; clear without submitting.")
            return _reset_sell_campaign(None)
        min_row_px = _finite_float((minimum_row or {}).get("price"), positive=True)
        if (
            min_row_px is not None
            and not _has_pending_campaign(state)
            and _is_corrupted_sell_trough(trough_price, trough_ts, min_row_px, current_price)
        ):
            print(
                f"[WARNING] [{symbol}] Resetting corrupted SELL campaign trough_price={trough_price:.8f} "
                f"(validated window_min={min_row_px:.8f}, current={current_price:.8f})."
            )
            return _reset_sell_campaign(None)
        growth = ((current_price - trough_price) / trough_price) * 100.0
        if growth <= SELL_REARM_GROWTH_PERCENT:
            print(
                f"[{symbol}] SELL campaign rearmed: growth={growth:.4f}% <= "
                f"{SELL_REARM_GROWTH_PERCENT:.4f}% against frozen trough "
                f"{trough_price:.8f}.")
            return _reset_sell_campaign(growth)
        state = _reconcile_campaign_initial_amount(
            symbol, "sell", state, free_qty, open_sells_loader=_get_open_sells
        )
    else:
        trough_price = _finite_float(minimum_row.get("price"), positive=True)
        trough_ts = _finite_float(minimum_row.get("timestamp"), positive=True)
        if trough_price is None or trough_ts is None:
            return None, {}, None
        growth = ((current_price - trough_price) / trough_price) * 100.0
        if growth < min_trigger_thr:
            _rtrade_sell_block_since.pop(symbol, None)
            return None, {}, growth
        state = {
            "trough_price": trough_price,
            "trough_ts": trough_ts,
            "initial_qty": free_qty,
            "created_at": time.time(),
            "completed_tiers": [],
            "filled_qty_by_tier": {},
            "total_filled_qty": 0.0,
            "attempts_by_tier": {},
            "terminal_orders": [],
        }
    completed = _completed_tier_values(state)
    initial_qty = _finite_float(state.get("initial_qty"), positive=True)
    raw_filled_by_tier = state.get("filled_qty_by_tier")
    filled_by_tier = dict(raw_filled_by_tier) if isinstance(raw_filled_by_tier, dict) else {}
    due_tier = _first_due_sell_tier(
        growth, completed, sell_mult, portfolio_growth_pct, vol_mult, active_tiers=active_tiers
    )
    if due_tier is None:
        _rtrade_sell_block_since.pop(symbol, None)
        return None, ({} if new_campaign else state), growth
    due_threshold, due_allocation = due_tier
    due_unfilled_qty = max(
        0.0,
        (initial_qty or 0.0) * due_allocation
        - safe_nonneg_float(filled_by_tier.get(_tier_key(due_threshold)), default=0.0),
    )
    open_sells = _get_open_sells()
    if open_sells is None:
        print(f"[{symbol}] Cannot verify open SELL orders; do not submit tier.")
        return None, ({} if new_campaign else state), growth
    needs_locked_balance = new_campaign or (free_qty + 1e-12 < due_unfilled_qty)
    open_sells = _filter_blocking_sells(open_sells, needs_locked=needs_locked_balance)
    if not open_sells:
        _rtrade_sell_block_since.pop(symbol, None)
    had_open_sells = bool(open_sells)
    if open_sells and _can_preempt_sell():
        open_sells = _resolve_blocking_sell_orders(
            symbol, open_sells, canceled_orders_out=canceled_orders_out
        )
    if open_sells:
        print(
            f"[{symbol}] Existing SELL order blocks AssetGuardian tier: "
            f"{[extract_order_id(order) for order in open_sells]}")
        return None, ({} if new_campaign else state), growth
    if had_open_sells:
        refreshed_free = _refresh_free_base()
        if refreshed_free is not None and refreshed_free > free_qty:
            free_qty = refreshed_free
            if new_campaign:
                state["initial_qty"] = free_qty
            else:
                state = _reconcile_campaign_initial_amount(symbol, "sell", state, free_qty)
            initial_qty = _finite_float(state.get("initial_qty"), positive=True)
    if new_campaign:
        _save_symbol_campaign(symbol, "sell", state)

    tier_target_qty = initial_qty * due_allocation
    already_filled = safe_nonneg_float(
        filled_by_tier.get(_tier_key(due_threshold)), default=0.0
    )
    unfilled_tier_qty = max(0.0, tier_target_qty - already_filled)
    remaining_qty = min(free_qty, unfilled_tier_qty)
    if remaining_qty <= max(1e-12, tier_target_qty * 0.02):
        _complete_campaign_tier(symbol, "sell", state, due_threshold)
        return None, _symbol_campaign(symbol, "sell"), growth
    return (due_threshold, due_allocation, remaining_qty), state, growth


def _safe_tier_attempt(attempts_map, key):
    raw_val = attempts_map.get(key) if isinstance(attempts_map, dict) else None
    prior = exact_nonneg_int(raw_val, default=None)
    return 1 if prior is None else prior + 1


def _submit_campaign_tier(
    symbol, kind, state, threshold, requested_qty, requested_price, metadata, submit_fn,
):
    side_label = kind.upper()
    intent_kind = f"ASSET_GUARDIAN_{side_label}_TIER"
    refusal_reason = f"{kind}_not_accepted"
    key = _tier_key(threshold)
    attempt = _safe_tier_attempt(state.get("attempts_by_tier"), key)
    client_order_id = _ag_client_order_id(kind, symbol, state, threshold, attempt)
    raw_attempts = state.get("attempts_by_tier")
    state = {**state, "attempts_by_tier": {**(raw_attempts if isinstance(raw_attempts, dict) else {}), key: attempt}}
    last_ref = state.get("last_refusal")
    if isinstance(last_ref, dict) and last_ref.get("intent_id") == client_order_id and last_ref.get("reason"):
        ref_map = getattr(ORDER_LIFECYCLE, "_last_refusal_by_intent", None)
        if isinstance(ref_map, dict):
            ref_map[(str(client_order_id), str(symbol), str(side_label), str(intent_kind))] = str(last_ref["reason"])
    state.pop("last_refusal", None)
    intent = ORDER_LIFECYCLE.new_intent(
        intent_id=client_order_id, client_order_id=client_order_id, symbol=symbol,
        side=side_label, requested_qty=requested_qty, requested_price=requested_price,
        kind=intent_kind, attempt=attempt, metadata=metadata,
    )
    box, persist = _campaign_pending_persistor(symbol, kind, state)
    _LAST_SUBMIT_OUTCOME.clear()

    def _submit():
        resp = submit_fn(client_order_id)
        if resp is False and not _LAST_SUBMIT_OUTCOME.get("state"):
            _LAST_SUBMIT_OUTCOME.update(accepted=False, state="refused", reason=refusal_reason)
        return propagate_submission_refusal(resp, _LAST_SUBMIT_OUTCOME)

    try:
        result = ORDER_LIFECYCLE.submit(intent, persist=persist, submit=_submit)
    except Exception as exc:
        print(f"[{symbol}] {side_label} tracked submit failed before/around submit: {exc}")
        return False
    if result.outcome == "refused":
        persist(None)
        _reuse_unaccepted_attempt(symbol, kind, box["state"], result.intent)
        return False
    # Known order ID means venue acceptance was observed.  An ambiguous response
    # remains pending and is recovered by client ID in a later evaluator cycle.
    return result.order_known


def _submit_sell_tier(symbol, current_price, tier, state):
    threshold, allocation, qty = tier
    return _submit_campaign_tier(
        symbol, "sell", state, threshold, qty, current_price,
        {"threshold": threshold, "allocation": allocation},
        lambda client_order_id: sell_asset(
            symbol, qty=qty, current_price=current_price,
            client_order_id=client_order_id, tier_threshold=threshold,
        ),
    )


def _submit_buy_tier(symbol, current_price, cash_amount, tier, state):
    threshold, allocation = tier
    respect_guard = GUARDED_BUY_THRESHOLD is not None and threshold == GUARDED_BUY_THRESHOLD
    cash_f = _finite_float(cash_amount, positive=True)
    px_f = _finite_float(current_price, positive=True)
    if cash_f is None or px_f is None:
        return False
    return _submit_campaign_tier(
        symbol, "buy", state, threshold, cash_f / px_f, current_price,
        {"threshold": threshold, "allocation": allocation, "requested_cash": cash_amount},
        lambda client_order_id: buy_with_all_cash(
            buy_symbol=symbol, cash_amount=cash_amount,
            client_order_id=client_order_id, respect_guard=respect_guard,
        ),
    )


def _next_check_seconds():
    if any(state.get("pending_tier") for state in _last_evaluation.values()):
        return min(CHECK_INTERVAL_SECONDS, ACTIVE_TRIGGER_SECONDS)
    root = _load_state_root()
    if any(
        any(isinstance(row.get(kind, {}).get("pending"), dict) for kind in ("buy", "sell"))
        for row in root["symbols"].values() if isinstance(row, dict)
    ) or any(_rtrade_sell_block_since.values()):
        return min(CHECK_INTERVAL_SECONDS, ACTIVE_TRIGGER_SECONDS)
    sell_mult, buy_mult, _ = _p3b_threshold_multipliers()
    for symbol, evaluation in _last_evaluation.items():
        vol_mult = _symbol_volatility_mult(symbol)
        drawdown = evaluation.get("drawdown")
        row = casefold_get(root["symbols"], symbol, {})
        buy_completed = _completed_tier_values(row.get("buy", {}))
        buy_next = next(
            (_trigger_threshold_for_buy_tier(threshold, buy_mult, volatility_mult=vol_mult)
             for threshold, _ in BUY_TIERS if threshold not in buy_completed),
            None,
        )
        if drawdown is not None and buy_next is not None and buy_next - max(0.0, -drawdown) <= NEAR_TRIGGER_DISTANCE_PCT:
            return min(CHECK_INTERVAL_SECONDS, NEAR_TRIGGER_SECONDS)
        growth = evaluation.get("growth")
        pg = evaluation.get("portfolio_growth")
        sell_completed = _completed_tier_values(row.get("sell", {}))
        active_tiers = _effective_sell_tiers(pg, sell_mult=sell_mult)
        sell_next = next(
            (_trigger_threshold_for_sell_tier(threshold, sell_mult, pg, volatility_mult=vol_mult)
             for threshold, _ in active_tiers if threshold not in sell_completed),
            None,
        )
        if growth is not None and sell_next is not None and sell_next - max(0.0, growth) <= NEAR_TRIGGER_DISTANCE_PCT:
            return min(CHECK_INTERVAL_SECONDS, NEAR_TRIGGER_SECONDS)
    return CHECK_INTERVAL_SECONDS


@_cycle_io_scope()
def evaluate_symbol(symbol, minutes_back=ASSET_REFERENCE_MINUTES_BACK_DEFAULT):
    """Evaluate and possibly submit exactly one asset-specific SELL or BUY."""
    symbol = str(symbol or "").strip().upper()
    evaluation = _last_evaluation.setdefault(
        symbol, {"growth": None, "drawdown": None, "pending_tier": False, "volatility_mult": 1.0}
    )
    evaluation.update(growth=None, drawdown=None, pending_tier=False)
    if symbol not in TRACKED_SYMBOLS:
        return _log_symbol_and_return(symbol, "Invalid AssetGuardian symbol; skip evaluation.")

    for kind in ("sell", "buy"):
        campaign = _symbol_campaign(symbol, kind)
        if not _has_pending_campaign(campaign):
            continue
        _campaign, pending_outcome = _reconcile_pending_campaign(symbol, kind, campaign)
        if pending_outcome == "active":
            evaluation["pending_tier"] = True
            return False
        if pending_outcome == "terminal":
            return False

    current_price = _finite_float(api.get_current_price(symbol), positive=True)
    if current_price is None:
        return _log_symbol_and_return(symbol, "Current price unavailable; skip evaluation.")
    extrema = _get_symbol_window_extrema(symbol, current_price, minutes_back=minutes_back)
    if not extrema:
        return _log_symbol_and_return(symbol, f"No per-asset baseline for the last {minutes_back}m.")

    minimum_row, maximum_row = extrema
    minimum_price = _finite_float(minimum_row.get("price"), positive=True)
    maximum_price = _finite_float(maximum_row.get("price"), positive=True)
    if minimum_price is None or maximum_price is None:
        return _log_symbol_and_return(symbol, "Invalid per-asset baseline.")

    rolling_growth_percent = ((current_price - minimum_price) / minimum_price) * 100.0
    drawdown_percent = ((current_price - maximum_price) / maximum_price) * 100.0
    evaluation["drawdown"] = drawdown_percent
    existing_buy_campaign = _symbol_campaign(symbol, "buy")
    existing_sell_campaign = _symbol_campaign(symbol, "sell")
    eval_now_ts = time.time() * 1.0
    eff_min_back = _finite_float(minutes_back, positive=True) or (
        safe_pos_float(ASSET_REFERENCE_MINUTES_BACK_DEFAULT, default=1440.0) or 1440.0
    )
    window_start_ts = eval_now_ts - eff_min_back * 60.0
    if existing_buy_campaign and not _has_pending_campaign(existing_buy_campaign):
        if _is_corrupted_buy_peak(
            existing_buy_campaign.get("peak_price"),
            existing_buy_campaign.get("peak_ts"),
            maximum_price,
            current_price,
            window_start_ts=window_start_ts,
            now_ts=eval_now_ts,
        ):
            print(
                f"[WARNING] [{symbol}] Auto-healing corrupted persisted BUY campaign "
                f"(peak_price={existing_buy_campaign.get('peak_price')}, "
                f"validated window_max={maximum_price:.8f}, current={current_price:.8f})."
            )
            _save_symbol_campaign(symbol, "buy", {})
            existing_buy_campaign = {}
    if existing_sell_campaign and not _has_pending_campaign(existing_sell_campaign):
        if _is_corrupted_sell_trough(
            existing_sell_campaign.get("trough_price"),
            existing_sell_campaign.get("trough_ts"),
            minimum_price,
            current_price,
            window_start_ts=window_start_ts,
            now_ts=eval_now_ts,
        ):
            print(
                f"[WARNING] [{symbol}] Auto-healing corrupted persisted SELL campaign "
                f"(trough_price={existing_sell_campaign.get('trough_price')}, "
                f"validated window_min={minimum_price:.8f}, current={current_price:.8f})."
            )
            _save_symbol_campaign(symbol, "sell", {})
            existing_sell_campaign = {}
    if drawdown_percent > -RECOVERY_RESET_PERCENT and existing_buy_campaign:
        print(f"[{symbol}] Drawdown recovered below {RECOVERY_RESET_PERCENT:.2f}%; rearm BUY campaign.")
        _save_symbol_campaign(symbol, "buy", {})

    provider = mkt.provider_by_name("binance")
    try:
        base_asset, _ = resolve_assets(symbol)
        free_base_qty = provider.free_balance(base_asset) if provider is not None else None
    except Exception as exc:
        print(f"[{symbol}] SELL balance unavailable; skip SELL fail-closed: {exc}")
        free_base_qty = None
    portfolio_growth_pct = _cycle_portfolio_growth_percent(minutes_back=minutes_back)
    evaluation["portfolio_growth"] = portfolio_growth_pct
    p3b_assessment = _cycle_p3b_sentiment()
    sell_mult, buy_mult, _ = _p3b_threshold_multipliers(p3b_assessment)
    preempted_sell_orders: list[dict] = []
    sell_tier, sell_state, effective_growth = _sell_campaign_tier(
        symbol, current_price, minimum_row, free_base_qty, provider,
        portfolio_growth_pct=portfolio_growth_pct, p3b_assessment=p3b_assessment,
        canceled_orders_out=preempted_sell_orders,
    )
    if sell_tier is None:
        if preempted_sell_orders:
            _restore_preempted_orders(symbol, preempted_sell_orders, "assetguardian_sell_no_tier")
        cand_sell_g = effective_growth if effective_growth is not None else rolling_growth_percent
        due_sell_pair = _first_due_sell_tier(
            cand_sell_g, _completed_tier_values(sell_state), sell_mult, portfolio_growth_pct, _symbol_volatility_mult(symbol)
        )
        if (
            due_sell_pair is not None
            and provider is not None
            and has_preemptible_open_orders(provider, symbol, "SELL", "assetguardian")
            and _is_short_trend_waiting(symbol, "SELL")
        ):
            evaluation["pending_tier"] = True
            if isinstance(sell_state, dict) and sell_state:
                _, sell_state = _trend_defer_ready(symbol, "sell", "SELL", due_sell_pair[0], sell_state)
        else:
            sell_state = _clear_campaign_trend_defer(symbol, "sell", sell_state)
    evaluation["growth"] = rolling_growth_percent if effective_growth is None else effective_growth
    frozen_trough = _finite_float(sell_state.get("trough_price"), positive=True)
    pg_str = f"{portfolio_growth_pct:.4f}%" if portfolio_growth_pct is not None else "n/a"
    p3b_conf = clamp_float(p3b_assessment.get("confidence"), 0.0, 1.0, default=0.0) if isinstance(p3b_assessment, dict) else 0.0
    p3b_str = (
        f"{p3b_assessment.get('recommended_action')}/{p3b_assessment.get('market_bias')}"
        f"(conf={p3b_conf:.2f},sell_x={sell_mult:.3f},buy_x={buy_mult:.3f})"
        if isinstance(p3b_assessment, dict) else "none(1.0x)"
    )
    default_min_back = safe_pos_float(ASSET_REFERENCE_MINUTES_BACK_DEFAULT, default=1440.0) or 1440.0
    minutes_back_val = _finite_float(minutes_back, positive=True) or default_min_back
    vol_mult = _symbol_volatility_mult(symbol)
    print(
        f"[{symbol}] price={current_price:.8f}, window_min/max="
        f"{minimum_price:.8f}/{maximum_price:.8f}, minutes_back={minutes_back_val:.1f}, "
        f"rolling_growth_from_min={rolling_growth_percent:.4f}%, "
        f"sell_growth={evaluation['growth']:.4f}%, portfolio_growth={pg_str}, "
        f"p3b={p3b_str}, vol_x={vol_mult:.3f}, "
        f"frozen_sell_trough={frozen_trough}, drawdown_from_max={drawdown_percent:.4f}%"
    )

    if sell_tier is not None:
        threshold, allocation, qty = sell_tier
        eff_sell_thr = _trigger_threshold_for_sell_tier(threshold, sell_mult, portfolio_growth_pct, volatility_mult=vol_mult)
        print(
            f"[{symbol}] SELL tier +{threshold:.2f}% (trigger +{eff_sell_thr:.2f}%) "
            f"allocation={allocation:.3f} target_remaining={qty:.8f} base."
        )
        evaluation["pending_tier"] = True
        ready, sell_state = _trend_defer_ready(symbol, "sell", "SELL", threshold, sell_state)
        if not ready:
            _restore_preempted_orders(symbol, preempted_sell_orders, "assetguardian_sell_trend_deferred")
            return False
        if _submit_sell_tier(symbol, current_price=current_price, tier=sell_tier, state=sell_state):
            evaluation["pending_tier"] = _has_pending_campaign(_symbol_campaign(symbol, "sell"))
            return True
        if _LAST_SUBMIT_OUTCOME.get("state") != "unknown":
            _restore_preempted_orders(symbol, preempted_sell_orders, "assetguardian_sell_refused")
        return False

    first_buy_trigger = _trigger_threshold_for_buy_tier(BUY_TIERS[0][0], buy_mult, volatility_mult=vol_mult)
    if drawdown_percent > -first_buy_trigger:
        _clear_campaign_trend_defer(symbol, "buy", _symbol_campaign(symbol, "buy"))
        return False
    print(f"[{symbol}] Drawdown threshold reached ({drawdown_percent:.4f}% <= -{first_buy_trigger:.4f}%). Checking staged BUY...")
    _, quote_asset = resolve_assets(symbol)
    free_cash = _finite_float(provider.free_balance(quote_asset), positive=True) if provider else None
    preempted_buy_orders: list[dict] = []
    if provider is not None:
        existing_buy = _symbol_campaign(symbol, "buy")
        due_buy_tier = _first_due_buy_tier(
            abs(drawdown_percent), _completed_tier_values(existing_buy), buy_mult, vol_mult
        )
        due_rem_cash = (
            _buy_remaining_cash(existing_buy, due_buy_tier[0], due_buy_tier[1])[0]
            if due_buy_tier is not None and existing_buy
            else None
        )
        needs_locked_quote = (
            not existing_buy
            or free_cash is None
            or free_cash < 10.0
            or (due_rem_cash is not None and free_cash + 1e-9 < due_rem_cash)
        )
        if due_buy_tier is not None and needs_locked_quote and has_preemptible_open_orders(
            provider, symbol, "BUY", "assetguardian"
        ):
            if _is_short_trend_waiting(symbol, "BUY"):
                evaluation["pending_tier"] = True
                if existing_buy:
                    _trend_defer_ready(symbol, "buy", "BUY", due_buy_tier[0], existing_buy)
                return False
            if preempt_lower_priority_open_orders(
                provider,
                symbol,
                "BUY",
                "assetguardian",
                cancel_fn=lambda oid: mkt.cancel_order(symbol, oid, provider_name="binance"),
                canceled_orders_out=preempted_buy_orders,
                resolve_retry_fn=resolve_accepted_order,
            ):
                free_cash = _finite_float(provider.free_balance(quote_asset), positive=True)
    if free_cash is None:
        _restore_preempted_orders(symbol, preempted_buy_orders, "assetguardian_buy_no_cash")
        return _log_symbol_and_return(symbol, f"No valid {quote_asset} balance for staged BUY.")
    tier, campaign = _campaign_tier(symbol, abs(drawdown_percent), maximum_row, free_cash, p3b_assessment=p3b_assessment)
    if tier is None:
        _restore_preempted_orders(symbol, preempted_buy_orders, "assetguardian_buy_no_tier")
        return _log_symbol_and_return(symbol, "No uncompleted BUY tier at current drawdown.")
    threshold, allocation = tier
    evaluation["pending_tier"] = True
    if _finite_float(campaign.get("initial_cash"), positive=True) is None:
        print(f"[{symbol}] Invalid campaign initial cash; reset fail-closed.")
        _save_symbol_campaign(symbol, "buy", {})
        _restore_preempted_orders(symbol, preempted_buy_orders, "assetguardian_buy_invalid_campaign")
        return False
    cash_amount, target_cash, prior_spent = _buy_remaining_cash(campaign, threshold, allocation)
    if cash_amount is None:
        _restore_preempted_orders(symbol, preempted_buy_orders, "assetguardian_buy_invalid_budget")
        return _log_symbol_and_return(symbol, "Invalid BUY tier budget; preserve fail-closed.")
    if cash_amount <= max(1e-9, target_cash * 0.02):
        _complete_campaign_tier(symbol, "buy", campaign, threshold)
        print(f"[{symbol}] BUY tier -{threshold:.2f}% already funded by terminal partials ({prior_spent:.6f} quote); mark complete.")
        _restore_preempted_orders(symbol, preempted_buy_orders, "assetguardian_buy_already_funded")
        return False
    print(f"[{symbol}] BUY tier -{threshold:.2f}% allocation={allocation:.3f} cash_remaining={cash_amount:.6f}/{target_cash:.6f} USDC")
    ready, campaign = _trend_defer_ready(symbol, "buy", "BUY", threshold, campaign)
    if not ready:
        _restore_preempted_orders(symbol, preempted_buy_orders, "assetguardian_buy_trend_deferred")
        return False
    if _submit_buy_tier(symbol, current_price, cash_amount, tier, campaign):
        evaluation["pending_tier"] = True
        return True
    if _LAST_SUBMIT_OUTCOME.get("state") != "unknown":
        _restore_preempted_orders(symbol, preempted_buy_orders, "assetguardian_buy_refused")
    return False


_STOP_EVENT = threading.Event()


@_cycle_io_scope()
def evaluate_and_maybe_sell_or_buy(
    minutes_back=ASSET_REFERENCE_MINUTES_BACK_DEFAULT,
    symbols=None,
):
    """Evaluate configured assets independently, accepting at most one order/cycle."""
    selected = tuple(symbols) if symbols is not None else TRACKED_SYMBOLS
    for symbol in selected:
        if _STOP_EVENT.is_set():
            break
        if evaluate_symbol(symbol, minutes_back=minutes_back):
            return True
    return False


def run_forever():
    _STOP_EVENT.clear()
    for _sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(_sig, lambda _signum, _frame: _STOP_EVENT.set())
        except Exception:
            pass
    print(
        f" Started. check_interval={CHECK_INTERVAL_SECONDS}s, "
        f"sell_tiers={SELL_TIERS}, sell_rearm={SELL_REARM_GROWTH_PERCENT}%, "
        f"order_max_age={ORDER_MAX_AGE_SECONDS}s, "
        f"trend_defer_max={TREND_DEFER_MAX_SECONDS}s, "
        f"buy_tiers={BUY_TIERS}, "
        f"minutes_back={ASSET_REFERENCE_MINUTES_BACK_DEFAULT}m, "
        f"symbols={TRACKED_SYMBOLS}"
    )
    touch_module_heartbeat(
        "assetguardian",
        interval_sec=30.0,
        force=True,
        symbols=list(TRACKED_SYMBOLS),
        intervals={"check_interval": CHECK_INTERVAL_SECONDS},
    )
    try:
        while not _STOP_EVENT.is_set():
            with _cycle_io_scope():
                try:
                    evaluate_and_maybe_sell_or_buy(minutes_back=ASSET_REFERENCE_MINUTES_BACK_DEFAULT)
                except Exception as e:
                    print(f" Runtime ERROR: {e}")
                if _STOP_EVENT.is_set():
                    break
                sleep_seconds = _next_check_seconds()
            touch_module_heartbeat(
                "assetguardian",
                interval_sec=30.0,
                symbols=list(TRACKED_SYMBOLS),
                intervals={"check_interval": sleep_seconds},
            )
            # Flushing at the cycle boundary publishes all preceding lines when stdout
            # is redirected by flota_start.sh, preventing a quiet bot from appearing stale.
            print(f"[DEBUG] sleep {sleep_seconds}s before next cycle", flush=True)
            _STOP_EVENT.wait(safe_nonneg_float(sleep_seconds, default=CHECK_INTERVAL_SECONDS))
    finally:
        request_stop_price_ws_streams()
        CacheManagerInterface.shutdown_all_instances(timeout=0.8)
        stop_price_ws_streams(time.monotonic() + 1.0)


if __name__ == "__main__":
    run_forever()
