"""core/time_market.py -- Canonical epoch normalization, US RTH market hours, and price-freshness SLA."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta, timezone
from functools import lru_cache
import sys
import threading
import time as _time
from zoneinfo import ZoneInfo

from cachepaths import (
    _parse_bool_flag,
    has_equity_suffix,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
    safe_pos_int,
)
from core.config import _eff_root_config_path, optional_env, parse_dotenv, required_float_env
from core.math_coercion import _first_mapped, coerce_dict_maps

ET = ZoneInfo("America/New_York")
US_EQUITY_PREMARKET_OPEN_ET = dt_time(4, 0)
US_EQUITY_RTH_OPEN_ET = dt_time(9, 30)
US_EQUITY_RTH_CLOSE_ET = dt_time(16, 0)
US_EQUITY_POSTMARKET_CLOSE_ET = dt_time(20, 0)


def _active_clock_fn() -> object:
    for mod_name in ("botcore", "core.heartbeat_runtime"):
        mod = sys.modules.get(mod_name)
        rs = getattr(mod, "_RUNTIME_SERVICES", None) if mod is not None else None
        clk = getattr(rs, "clock_fn", None)
        if callable(clk):
            return clk
    return None


@lru_cache(maxsize=32)
def _us_market_holidays_for_year(year: int) -> dict[date, str]:
    """Return observed NYSE full-day market holidays for ``year`` mapped to holiday names."""

    def _nth_weekday(month: int, weekday: int, n: int) -> date:
        first = date(year, month, 1)
        offset = (weekday - first.weekday()) % 7
        return first + timedelta(days=offset + 7 * (n - 1))

    def _last_weekday(month: int, weekday: int) -> date:
        nxt = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
        last = nxt - timedelta(days=1)
        return last - timedelta(days=(last.weekday() - weekday) % 7)

    def _observed(d: date, *, allow_prev_year_friday: bool = True) -> date | None:
        if d.weekday() == 5:
            if not allow_prev_year_friday and d.month == 1 and d.day == 1:
                return None
            return d - timedelta(days=1)
        if d.weekday() == 6:
            return d + timedelta(days=1)
        return d

    # Anonymous Gregorian Easter algorithm (Meeus/Jones/Butcher) for Good Friday
    a = year % 19
    b, c = divmod(year, 100)
    d_q, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d_q - g + 15) % 30
    i_q, k = divmod(c, 4)
    l_v = (32 + 2 * e + 2 * i_q - h - k) % 7
    m = (a + 11 * h + 22 * l_v) // 451
    e_month, e_rem = divmod(h + l_v - 7 * m + 114, 31)
    good_friday = date(year, e_month, e_rem + 1) - timedelta(days=2)

    holidays: dict[date, str] = {}
    for raw_d, name, allow_prev in (
        (date(year, 1, 1), "New Year's Day", False),
        (date(year, 7, 4), "Independence Day", True),
        (date(year, 12, 25), "Christmas Day", True),
    ):
        if (obs := _observed(raw_d, allow_prev_year_friday=allow_prev)) is not None:
            holidays[obs] = name
    if year >= 2022:
        if (obs := _observed(date(year, 6, 19))) is not None:
            holidays[obs] = "Juneteenth National Independence Day"
    holidays[_nth_weekday(1, 0, 3)] = "Martin Luther King Jr. Day"
    holidays[_nth_weekday(2, 0, 3)] = "Washington's Birthday"
    holidays[good_friday] = "Good Friday"
    holidays[_last_weekday(5, 0)] = "Memorial Day"
    holidays[_nth_weekday(9, 0, 1)] = "Labor Day"
    holidays[_nth_weekday(11, 3, 4)] = "Thanksgiving Day"
    return holidays


def us_market_holiday_name(day: date | datetime) -> str | None:
    """Return the NYSE full-day holiday name for ``day`` (in ET date), or ``None`` if not a holiday."""
    d = day.date() if isinstance(day, datetime) else day
    if not isinstance(d, date):
        return None
    return _us_market_holidays_for_year(d.year).get(d)


def is_us_market_holiday(day: date | datetime) -> bool:
    """Return True when ``day`` is an observed NYSE full-day market holiday."""
    return us_market_holiday_name(day) is not None


def resolve_us_equity_session_state(now_ts: float | None = None) -> str:
    """Return canonical US equity session state ('PRE', 'REGULAR', 'POST', or 'CLOSED') in America/New_York."""
    ts = resolve_now_ts(now_ts)
    try:
        ny_dt = datetime.fromtimestamp(ts, tz=ET)
    except (ValueError, OverflowError, OSError):
        ny_dt = datetime.fromtimestamp(_time.time(), tz=ET)
    if ny_dt.weekday() >= 5 or is_us_market_holiday(ny_dt.date()):
        return "CLOSED"
    t_ny = ny_dt.time()
    if US_EQUITY_RTH_OPEN_ET <= t_ny < US_EQUITY_RTH_CLOSE_ET:
        return "REGULAR"
    if US_EQUITY_PREMARKET_OPEN_ET <= t_ny < US_EQUITY_RTH_OPEN_ET:
        return "PRE"
    if US_EQUITY_RTH_CLOSE_ET <= t_ny < US_EQUITY_POSTMARKET_CLOSE_ET:
        return "POST"
    return "CLOSED"


def is_us_rth_open(now_ts: float | None = None) -> bool:
    """Return True if US equities regular trading hours (09:30-16:00 America/New_York, Mon-Fri) are open."""
    return resolve_us_equity_session_state(now_ts) == "REGULAR"


def is_us_extended_hours_open(now_ts: float | None = None) -> bool:
    """Return True if US equities extended trading hours (04:00-09:30 or 16:00-20:00 America/New_York, Mon-Fri) are open."""
    return resolve_us_equity_session_state(now_ts) in ("PRE", "POST")


def timestampToTime(timestamp_ms) -> str:
    ts_ms = safe_nonneg_float(timestamp_ms, default=None)
    if ts_ms is None:
        return "N/A"
    try:
        return datetime.fromtimestamp(ts_ms / 1000.0).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError, OverflowError, OSError):
        return "N/A"


@lru_cache(maxsize=256)
def timeToHMS(timestamp_sec) -> str:
    ts_f = normalize_epoch_sec(timestamp_sec, default=None, allow_zero=True)
    if ts_f is None:
        return "N/A"
    try:
        return datetime.fromtimestamp(ts_f).strftime("%H:%M:%S")
    except (TypeError, ValueError, OverflowError, OSError):
        return "N/A"


def _bind_seconds_converter(divisor: float):
    return lambda max_age_seconds, _div=divisor: safe_nonneg_float(max_age_seconds, default=0.0) / _div


secondsToDays = _bind_seconds_converter(86400.0)
secondsToHours = _bind_seconds_converter(3600.0)


def parse_iso_datetime_utc(raw: str) -> datetime | None:
    """Parse an ISO-8601 string (supporting trailing ``Z``/``z``) into a timezone-aware UTC ``datetime``."""
    try:
        iso_s = raw[:-1] + "+00:00" if raw.endswith(("Z", "z")) else raw
        dt = datetime.fromisoformat(iso_s)
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
    except Exception:
        return None


def normalize_epoch_sec(
    val: object, default: float | None = None, *, allow_zero: bool = False, allow_iso: bool = False
) -> float | None:
    """Normalize a timestamp in seconds, milliseconds (>= 1e11), or optional ISO-8601 string into epoch seconds."""
    f = safe_nonneg_float(val, default=None) if allow_zero else safe_pos_float(val, default=None)
    if f is None and allow_iso and isinstance(val, str):
        s = val.strip()
        if len(s) >= 10 and ("-" in s or "T" in s):
            if (dt := parse_iso_datetime_utc(s)) is not None:
                ts_s = dt.timestamp()
                if (ts_s >= 0.0) if allow_zero else (ts_s > 0.0):
                    f = ts_s
    if f is None:
        return default
    return f / 1000.0 if f >= 1e11 else f


def first_epoch_sec(
    *values: object, default: float | None = None, allow_iso: bool = False
) -> float | None:
    """Return the first strictly positive epoch-seconds timestamp among ``values``, skipping None/0/'0'/NaN/Inf/bool."""
    for val in values:
        if (ts_s := normalize_epoch_sec(val, default=None, allow_zero=False, allow_iso=allow_iso)) is not None:
            return ts_s
    return default


def parse_epoch_price_pair(
    item: object, *, allow_zero_ts: bool = True
) -> tuple[float, float] | None:
    """Parse a 2-element ``(ts, price)`` list/tuple into ``(epoch_sec, positive_price)`` or ``None``."""
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        return None
    t = normalize_epoch_sec(item[0], default=None, allow_zero=allow_zero_ts)
    p = safe_pos_float(item[1], default=None)
    return (t, p) if (t is not None and p is not None) else None


def safe_nonneg_ts(val: object, default: float = 0.0) -> float:
    """Safely coerce a timestamp to non-negative epoch seconds, converting ms (>= 1e11)."""
    res = normalize_epoch_sec(val, default=default, allow_zero=True)
    return default if res is None else res


def resolve_now_ts(now_ts: object = None, *, allow_zero: bool = False) -> float:
    """Return ``now_ts`` normalized to epoch seconds (converting ms >= 1e11), or the active runtime clock."""
    f = normalize_epoch_sec(now_ts, default=None, allow_zero=allow_zero)
    if f is not None:
        return f
    clk = _active_clock_fn()
    return safe_float(clk(), default=_time.time()) if callable(clk) else _time.time()


def resolve_now_ms(now_ms: object = None) -> int:
    """Return ``now_ms`` as a positive integer millisecond timestamp, or ``int(resolve_now_ts() * 1000)``."""
    parsed = safe_pos_int(now_ms, default=None)
    return parsed if parsed is not None else round(resolve_now_ts() * 1000.0)


def resolve_remaining_timeout_sec(
    deadline_mono: float | None,
    max_timeout: float,
    *,
    min_timeout: float = 0.0,
    now_mono: float | None = None,
) -> float | None:
    """Return remaining seconds until ``deadline_mono`` capped at ``max_timeout``, or ``None`` if below ``min_timeout``."""
    cap = safe_pos_float(max_timeout, default=None)
    if cap is None:
        return None
    if deadline_mono is None:
        return cap
    dl = safe_float(deadline_mono, default=None)
    if dl is None:
        return cap
    cur = safe_float(now_mono, default=None)
    if cur is None:
        cur = _time.monotonic()
    rem = dl - cur
    floor = safe_nonneg_float(min_timeout, default=0.0) or 0.0
    if rem <= 0.0 or (floor > 0.0 and rem < floor):
        return None
    return min(cap, rem)


_TIMESTAMP_KEYS = (
    "timestamp", "time", "updateTime", "transactTime", "opentm", "closetm", "workingTime",
    "filledAt", "dateExecuted", "dateModified", "createdAt", "dateCreated", "T", "E", "ts", "ts_ms",
)


def _parse_epoch_or_iso_s(value: object) -> float | None:
    numeric = safe_nonneg_float(value, default=None)
    return numeric if numeric is not None else normalize_epoch_sec(value, default=None, allow_zero=True, allow_iso=True)


def normalize_timestamp_ms(value: object, default: int | None = None) -> int | None:
    """Normalize a raw timestamp (in seconds, milliseconds, or ISO-8601 string) to integer milliseconds."""
    numeric = safe_nonneg_float(value, default=None)
    if numeric is None:
        numeric = _parse_epoch_or_iso_s(value)
    if numeric is None:
        return default
    if numeric == 0.0:
        return 0
    if 1_000_000_000.0 <= numeric < 100_000_000_000.0 or (
        0.0 < numeric < 1_000_000_000.0 and isinstance(value, float) and not numeric.is_integer()
    ):
        return round(numeric * 1000.0)
    return round(numeric)


def _ts_ms_to_s(ts_ms: float | int) -> float:
    return normalize_epoch_sec(ts_ms, default=0.0, allow_zero=True) or 0.0


def _is_fresh_obs_ts(obs_ts_s: float | None, now_s: float, max_age_s: float) -> bool:
    return obs_ts_s is not None and -5.0 <= (now_s - obs_ts_s) <= max_age_s


def extract_timestamp_ms(order: object, default: int | None = None) -> int | None:
    """Extract and normalize an order/fill/tick timestamp in milliseconds from any venue shape."""
    if isinstance(order, (list, tuple)):
        return normalize_timestamp_ms(order[0], default=default) if order else default
    pos_ms = _first_mapped(
        order,
        _TIMESTAMP_KEYS,
        lambda v: ms if (ms := normalize_timestamp_ms(v, default=None)) is not None and ms > 0 else None,
        None,
    )
    if pos_ms is not None:
        return pos_ms
    return _first_mapped(order, _TIMESTAMP_KEYS, lambda v: normalize_timestamp_ms(v, default=None), default)


def extract_timestamp_s(order: object, default: float | None = None) -> float | None:
    """Extract and normalize an order/fill/tick timestamp in seconds from any venue shape."""
    if isinstance(order, (list, tuple)):
        return normalize_epoch_sec(order[0], default=default, allow_zero=True, allow_iso=True) if order else default
    pos_s = _first_mapped(
        order,
        _TIMESTAMP_KEYS,
        lambda v: normalize_epoch_sec(v, default=None, allow_zero=False, allow_iso=True),
        None,
    )
    if pos_s is not None:
        return pos_s
    return _first_mapped(order, _TIMESTAMP_KEYS, lambda v: normalize_epoch_sec(v, allow_zero=True, allow_iso=True), default)


def is_cache_fresh(
    last_ts: object, now_ts: float, ttl_sec: object, *, force_refresh: object = False
) -> bool:
    """Return True if cached state at ``last_ts`` is still within ``ttl_sec`` at ``now_ts``."""
    if _parse_bool_flag(force_refresh, False):
        return False
    now_s = safe_nonneg_float(now_ts, None)
    if now_s is None:
        return False
    lt = safe_nonneg_float(last_ts, 0.0)
    ttl = safe_nonneg_float(ttl_sec, 0.0)
    return lt > 0.0 and 0.0 <= (now_s - lt) < ttl


def is_interval_due(
    current_ts: float, last_ts: object, interval_sec: object, *, force: object = False
) -> bool:
    """Return True if ``force`` is truthy or ``interval_sec`` has elapsed since ``last_ts``."""
    if _parse_bool_flag(force, False):
        return True
    cur_s = safe_nonneg_float(current_ts, None)
    if cur_s is None:
        return True
    lt = safe_nonneg_float(last_ts, 0.0)
    itv = safe_nonneg_float(interval_sec, 0.0)
    return not (0.0 <= (cur_s - lt) < itv)


def _is_safe_dict_key(val: object) -> bool:
    return isinstance(val, (str, int, float, bytes, type(None))) or (
        isinstance(val, tuple) and all(_is_safe_dict_key(x) for x in val)
    )


def check_cooldown_gate(
    cooldown_map: dict, key: object, now_ts: float, cooldown_sec: float = 1800.0, max_entries: int = 1000
) -> bool:
    """Return True and record ``now_ts`` if ``key`` is not within ``cooldown_sec``."""
    if not isinstance(cooldown_map, dict):
        return True
    now_f = safe_float(now_ts, None)
    if now_f is None:
        return True
    cd_f = safe_nonneg_float(cooldown_sec, 0.0) or 0.0
    map_key = key if _is_safe_dict_key(key) else str(key)
    raw_last = cooldown_map.get(map_key)
    last_f = safe_float(raw_last, None) if raw_last is not None else None
    if last_f is not None and 0.0 <= (now_f - last_f) < cd_f:
        return False
    cooldown_map[map_key] = now_f
    cap = safe_pos_int(max_entries, 1000) or 1000
    while len(cooldown_map) > cap:
        cooldown_map.pop(next(iter(cooldown_map)), None)
    return True


def extract_current_price_item(item: object) -> float | None:
    """Extract the latest positive finite price from any ``cache_current_price.json`` item format."""
    if item is None or isinstance(item, bool):
        return None
    if isinstance(item, (int, float, str)):
        return safe_pos_float(item, None)
    if isinstance(item, dict):
        return safe_pos_float(item.get("price") if item.get("price") is not None else item.get("c"), None)
    if isinstance(item, (list, tuple)):
        for entry in reversed(item):
            target = entry[1] if isinstance(entry, (list, tuple)) and len(entry) >= 2 else entry
            if (px_f := safe_pos_float(target, None)) is not None:
                return px_f
    return None


def _load_price_freshness_policy() -> tuple[float, float]:
    """Load and validate real-time price cadence and freshness SLA from root ``config.env``."""
    cfg_d = parse_dotenv(_eff_root_config_path())
    poll_raw = optional_env("CM_CURRENT_PRICE_POLL_SEC") or cfg_d.get("CM_CURRENT_PRICE_POLL_SEC")
    max_age_raw = optional_env("CM_PRICE_FRESHNESS_MAX_AGE_SEC") or cfg_d.get("CM_PRICE_FRESHNESS_MAX_AGE_SEC")
    poll_sec = required_float_env("CM_CURRENT_PRICE_POLL_SEC", {"CM_CURRENT_PRICE_POLL_SEC": poll_raw})
    max_age_sec = required_float_env("CM_PRICE_FRESHNESS_MAX_AGE_SEC", {"CM_PRICE_FRESHNESS_MAX_AGE_SEC": max_age_raw})
    if poll_sec <= 0.0:
        raise ValueError("CM_CURRENT_PRICE_POLL_SEC must be positive")
    if max_age_sec <= 0.0:
        raise ValueError("CM_PRICE_FRESHNESS_MAX_AGE_SEC must be positive")
    if poll_sec * 2.0 >= max_age_sec:
        raise ValueError(
            "CM_CURRENT_PRICE_POLL_SEC * 2.0 must be strictly less than CM_PRICE_FRESHNESS_MAX_AGE_SEC"
        )
    return poll_sec, max_age_sec


CM_CURRENT_PRICE_POLL_SEC, CM_PRICE_FRESHNESS_MAX_AGE_SEC = _load_price_freshness_policy()
PRICE_STREAM_POLL_INTERVAL_SEC: float = CM_CURRENT_PRICE_POLL_SEC
PRICE_FRESHNESS_MAX_AGE_SEC: float = CM_PRICE_FRESHNESS_MAX_AGE_SEC
PRICE_FRESHNESS_MAX_AGE_MS: int = round(PRICE_FRESHNESS_MAX_AGE_SEC * 1000.0)
PRICE_WS_TIMEOUT_SEC: float = round(PRICE_FRESHNESS_MAX_AGE_SEC * 0.5, 3)
PRICE_FALLBACK_HTTP_DEADLINE_SEC: float = round(PRICE_FRESHNESS_MAX_AGE_SEC * (2.0 / 3.0), 3)
SHORT_TREND_FRESHNESS_MAX_AGE_SEC: float = round(PRICE_FRESHNESS_MAX_AGE_SEC * 5.0, 3)


def extract_current_price_timestamp_sec(
    entry: object,
    *fallback_fetchtimes: object,
) -> tuple[float, bool]:
    """Extract ``(ts_epoch_sec, is_valid_structure)`` from a ``cache_current_price.json`` item."""
    if isinstance(entry, list):
        last_entry = entry[-1] if entry else None
        if not isinstance(last_entry, (list, tuple)) or len(last_entry) < 2:
            return 0.0, False
        return (normalize_epoch_sec(last_entry[0], default=0.0) or 0.0), True
    if isinstance(entry, dict):
        ts_val = (
            first_epoch_sec(
                entry.get("ts"),
                entry.get("timestamp"),
                entry.get("time"),
                *fallback_fetchtimes,
                default=0.0,
                allow_iso=True,
            )
            or 0.0
        )
        return ts_val, True
    ts_val = first_epoch_sec(*fallback_fetchtimes, default=0.0, allow_iso=True) or 0.0
    return ts_val, True


def _prices_diverged(px_a: object, px_b: object, threshold: float = 0.10) -> bool:
    """Return True when two positive prices diverge by more than ``threshold`` relative to their minimum."""
    a_f, b_f = safe_pos_float(px_a, default=0.0) or 0.0, safe_pos_float(px_b, default=0.0) or 0.0
    return (abs(a_f - b_f) / min(a_f, b_f) > threshold) if (a_f > 0.0 and b_f > 0.0) else False


@dataclass(frozen=True)
class TimedQuote:
    """Immutable timestamped price quote for freshness-aware cross-source comparisons."""

    price: float
    ts: float | None = None
    source: str = ""

    @classmethod
    def from_raw(cls, price: object, ts: object = None, *, source: str = "") -> "TimedQuote | None":
        px_f = safe_pos_float(price, default=None)
        if px_f is None:
            return None
        ts_f = normalize_epoch_sec(ts, default=None) if ts is not None else None
        return cls(price=px_f, ts=ts_f, source=source)

    def age_sec(self, now_ts: float | None) -> float | None:
        if self.ts is None or now_ts is None:
            return None
        return max(0.0, now_ts - self.ts)

    def is_fresh(self, now_ts: float | None, max_age_sec: float) -> bool:
        age = self.age_sec(now_ts)
        return age is None or age <= max_age_sec


def quote_spread_bps(quote_a: TimedQuote, quote_b: TimedQuote, *, decimals: int = 1) -> float:
    """Return signed basis-point difference ``((a - b) / b) * 10_000`` rounded to ``decimals``."""
    return round(((quote_a.price - quote_b.price) / quote_b.price) * 10000.0, decimals)


def resolve_aligned_quote_pair(
    primary_a: TimedQuote | None,
    primary_b: TimedQuote | None,
    fallback_a: TimedQuote | None = None,
    fallback_b: TimedQuote | None = None,
    *,
    now_ts: float | None = None,
    max_skew_sec: float = 2.0,
    max_age_sec: float = 10.0,
    max_fallback_divergence: float = 0.01,
    max_pair_divergence: float = 0.10,
) -> tuple[TimedQuote, TimedQuote] | None:
    """Select a temporally aligned quote pair across two sources without silently mixing stale fallbacks.

    Prefers ``(primary_a, primary_b)`` when temporally aligned and fresh; switches to
    ``(fallback_a, fallback_b)`` when primaries are skewed/stale *and* both fallbacks agree
    with their respective primaries within ``max_fallback_divergence``; finally guards the
    selected pair against cross-venue price divergence exceeding ``max_pair_divergence``.
    """
    eff_now = normalize_epoch_sec(now_ts, default=None) if now_ts is not None else None
    pa_ts = primary_a.ts if primary_a is not None else None
    pb_ts = primary_b.ts if primary_b is not None else None
    primaries_skewed = (
        pa_ts is not None
        and pb_ts is not None
        and abs(pa_ts - pb_ts) > max_skew_sec
    ) or (
        eff_now is not None
        and any(t is not None and (eff_now - t) > max_age_sec for t in (pa_ts, pb_ts))
    )
    fallback_consistent = (
        (primary_a is None or fallback_a is None or abs(fallback_a.price - primary_a.price) / primary_a.price <= max_fallback_divergence)
        and (primary_b is None or fallback_b is None or abs(fallback_b.price - primary_b.price) / primary_b.price <= max_fallback_divergence)
    )
    if primaries_skewed and fallback_a is not None and fallback_b is not None and fallback_consistent:
        q_a, q_b = fallback_a, fallback_b
    else:
        q_a = primary_a or fallback_a
        q_b = primary_b or fallback_b
    if q_a is None or q_b is None:
        return None
    if _prices_diverged(q_a.price, q_b.price, threshold=max_pair_divergence):
        return None
    return q_a, q_b


def summarize_current_price_cache_freshness(
    cp_payload: object,
    *,
    now_ts: float | None = None,
    max_age_sec: float = PRICE_FRESHNESS_MAX_AGE_SEC,
    market_open_fn: Callable[[str], bool] | None = None,
) -> dict[str, object]:
    """Evaluate ``cache_current_price.json`` freshness against the unified SSOT threshold."""
    now_s = resolve_now_ts(now_ts)
    eff_max_age = safe_pos_float(max_age_sec, default=PRICE_FRESHNESS_MAX_AGE_SEC) or PRICE_FRESHNESS_MAX_AGE_SEC
    (cp_d,) = coerce_dict_maps(cp_payload)
    has_items_envelope = isinstance(cp_d.get("items"), dict)
    raw_items, ft_map = coerce_dict_maps(cp_d.get("items") if has_items_envelope else cp_d, cp_d.get("fetchtime"))
    items_map = (
        raw_items
        if has_items_envelope
        else {k: v for k, v in raw_items.items() if k not in ("fetchtime", "rest_sync_cursor", "timestamp")}
    )
    ft_lower = {str(k).casefold(): v for k, v in ft_map.items()}

    all_ts: list[float] = [
        ts_ft
        for k, v in ft_map.items()
        if not isinstance(k, bool)
        and (ts_ft := (normalize_epoch_sec(v, default=0.0) or 0.0)) > 0.0
        and (now_s - ts_ft) >= -5.0
    ]
    open_ages: list[float] = []
    open_bounds: list[tuple[float, float]] = []
    any_open_invalid_or_missing_ts = False

    for sym_key, val in items_map.items():
        if isinstance(sym_key, bool):
            continue
        sym_str = str(sym_key or "").strip()
        if not sym_str:
            continue
        ft_val = ft_map.get(sym_key)
        if ft_val is None:
            ft_val = ft_lower.get(sym_str.casefold())
        ts_sec, valid_struct = extract_current_price_timestamp_sec(val, ft_val)
        if ts_sec > 0.0 and (now_s - ts_sec) >= -5.0:
            all_ts.append(ts_sec)
        px_val = extract_current_price_item(val) or 0.0
        is_eq = has_equity_suffix(sym_str)
        is_open = (
            (not is_eq or is_us_rth_open(now_s))
            if market_open_fn is None
            else _parse_bool_flag(market_open_fn(sym_str), default=False)
        )
        if not is_open:
            continue
        if not valid_struct or px_val <= 0.0 or ts_sec <= 0.0:
            any_open_invalid_or_missing_ts = True
            continue
        age = now_s - ts_sec
        sym_max_age = (
            max(eff_max_age, SHORT_TREND_FRESHNESS_MAX_AGE_SEC)
            if is_eq
            else eff_max_age
        )
        open_ages.append(age)
        open_bounds.append((age, sym_max_age))

    latest_ts = max(all_ts, default=0.0)
    freshest_age = round(max(0.0, now_s - latest_ts), 2) if latest_ts > 0.0 else None
    oldest_open_age = round(max(0.0, max(open_ages)), 2) if open_ages else None
    stream_max_age = min((bound for _, bound in open_bounds), default=eff_max_age)
    fresh_ok = bool(
        items_map
        and not any_open_invalid_or_missing_ts
        and freshest_age is not None
        and freshest_age < stream_max_age
        and (
            not open_bounds
            or all(-5.0 <= age < bound for age, bound in open_bounds)
        )
    )
    return {
        "symbols_count": len(items_map),
        "max_allowed_age_seconds": eff_max_age,
        "freshest_age_seconds": freshest_age,
        "oldest_open_age_seconds": oldest_open_age,
        "ws_realtime_fresh": fresh_ok,
    }


def resolve_sustained_sample_rate(
    cpm: object, symbol: str, *, min_span_sec: float = 5.0
) -> float | None:
    """Resolve sustained sample rate from a current-price manager or compatible test double."""
    if cpm is None:
        return None
    min_span_s = safe_nonneg_float(min_span_sec, default=5.0)
    if callable(getattr(type(cpm), "get_sustained_sample_rate", None)):
        return safe_pos_float(
            cpm.get_sustained_sample_rate(symbol, min_span_sec=min_span_s),
            default=None,
        )
    lock_obj = getattr(cpm, "lock", None)
    with lock_obj if hasattr(lock_obj, "__enter__") else threading.RLock():
        ts_map = getattr(cpm, "_update_timestamps", None)
        dq = ts_map.get(symbol) if isinstance(ts_map, dict) else None
        samples = list(dq) if dq else []
    if len(samples) < 2 or (samples[-1] - samples[0]) < min_span_s:
        return None
    sr_fn = getattr(cpm, "get_sample_rate", None)
    if callable(sr_fn):
        return safe_pos_float(sr_fn(symbol, fallback=None), default=None)
    return safe_pos_float(
        (samples[-1] - samples[0]) / (len(samples) - 1), default=None
    )


@dataclass(frozen=True)
class ClosedPriceSeries:
    """Completed closes with optional timestamps, volumes, and intrabar H/L/O arrays."""

    closes: tuple
    interval_min: int
    last_closed_at: float | None = None
    timestamps: tuple = ()
    volumes: tuple = ()
    highs: tuple = ()
    lows: tuple = ()
    opens: tuple = ()

    @classmethod
    def from_raw_candles(
        cls,
        raw_candles: object,
        interval_min: int,
        close_getter: object,
        ts_ms_getter: object,
        vol_getter: object = None,
        *,
        high_getter: object = None,
        low_getter: object = None,
        open_getter: object = None,
    ) -> "ClosedPriceSeries":
        completed = list(raw_candles or ())[:-1]  # type: ignore[arg-type]
        valid = [
            (item, c_f)
            for item in completed
            if (c_f := safe_pos_float(close_getter(item), default=None)) is not None  # type: ignore[operator]
        ]
        closes = tuple(c_f for _item, c_f in valid)
        ts_list = []
        ts_ok = bool(valid)
        for item, _ in valid:
            t_ms = safe_pos_float(ts_ms_getter(item), default=None)  # type: ignore[operator]
            if t_ms is None:
                ts_ok = False
                break
            ts_list.append(t_ms / 1000.0)
        timestamps = tuple(ts_list) if ts_ok else ()
        volumes = (
            tuple(safe_nonneg_float(vol_getter(item), default=0.0) or 0.0 for item, _ in valid)
            if callable(vol_getter) and valid
            else ()
        )

        def _opt_pos_series(getter: object) -> tuple:
            if not (callable(getter) and valid):
                return ()
            vals = tuple(safe_pos_float(getter(item), default=None) for item, _ in valid)
            return vals if any(v is not None for v in vals) else ()

        return cls(
            closes,
            safe_pos_int(interval_min, 1),
            timestamps[-1] if timestamps else None,
            timestamps,
            volumes,
            _opt_pos_series(high_getter),
            _opt_pos_series(low_getter),
            _opt_pos_series(open_getter),
        )
