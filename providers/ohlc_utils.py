"""Stateless OHLC candle, interval, Yahoo chart, and price-series utilities."""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence

from botcore import (
    ClosedPriceSeries,
    _parse_bool_flag,
    _prices_diverged as _yahoo_bar_prices_diverged,
    exact_int,
    exact_pos_int,
    normalize_epoch_sec,
    require_finite_float,
    require_int,
    safe_float,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
    safe_pos_int,
)

__all__ = [
    "ClosedPriceSeries",
    "OHLC_CASE_INSENSITIVE_INTERVAL_ALIASES",
    "OHLC_INTERVALS_MIN_TO_STR",
    "OHLC_INTERVALS_STR_TO_MIN",
    "ProviderError",
    "REQUIRED_FIELDS",
    "_CANDLE_INTERVALS",
    "align_previous_values",
    "build_price_history_points",
    "candle_interval",
    "candles_to_historical_points",
    "drop_incomplete_last_bar",
    "extract_trading_quote_pairs",
    "extract_yahoo_chart_latest_price",
    "extract_yahoo_chart_result_and_quote",
    "normalize_record",
    "normalize_yahoo_5d_chart_bars",
    "parse_ohlc_interval",
    "parse_ohlc_interval_minutes",
    "require_pos_ohlc_interval",
    "resolve_ohlc_time_window_ms",
    "series_to_historical_points",
    "validate_dataset",
    "validate_ohlc_bars",
    "validate_ohlc_candle_bar",
]


class ProviderError(Exception):
    """Venue-neutral error raised by strict strategy-execution adapters."""


OHLC_INTERVALS_MIN_TO_STR: dict[int, str] = {
    1: "1m", 3: "3m", 5: "5m", 15: "15m", 30: "30m",
    60: "1h", 120: "2h", 240: "4h", 360: "6h", 480: "8h", 720: "12h",
    1440: "1d", 4320: "3d", 10080: "1w", 43200: "1M",
}
OHLC_INTERVALS_STR_TO_MIN: dict[str, int] = {v: k for k, v in OHLC_INTERVALS_MIN_TO_STR.items()}
OHLC_CASE_INSENSITIVE_INTERVAL_ALIASES: dict[str, str] = {
    "1H": "1h", "2H": "2h", "4H": "4h", "6H": "6h", "8H": "8h", "12H": "12h",
    "1D": "1d", "3D": "3d", "1W": "1w",
}
_OHLC_UNIT_MULTIPLIERS: dict[str, int] = {"m": 1, "h": 60, "d": 1440}
_CANDLE_INTERVALS: dict[int, str] = {
    k: OHLC_INTERVALS_MIN_TO_STR[k] for k in (1, 5, 15, 60, 240, 1440)
}


def candle_interval(interval_min: int) -> str:
    """Return the canonical venue interval or reject an unsupported horizon."""
    minutes = exact_int(interval_min, default=None)
    if minutes is None:
        raise ProviderError(f"invalid candle interval: {interval_min!r}")
    if minutes not in _CANDLE_INTERVALS:
        raise ProviderError(f"unsupported candle interval: {interval_min!r} minutes")
    return _CANDLE_INTERVALS[minutes]


def require_pos_ohlc_interval(symbol: object, interval_min: Any) -> int:
    """Validate and return a positive integer OHLC interval in minutes or raise ``ProviderError``."""
    interval_i = exact_pos_int(interval_min, default=None)
    if interval_i is None:
        raise ProviderError(f"ohlc_series({symbol}): invalid interval_min={interval_min!r}")
    return interval_i


def build_price_history_points(
    rows: Any,
    extract_ts: Any,
    extract_px: Any,
    *,
    cutoff_s: Optional[float] = None,
    ts_in_seconds: bool = False,
) -> List[dict]:
    """Extract, filter, and sort ascending ``{'timestamp': ms, 'price': float}`` rows."""
    if not isinstance(rows, (list, tuple)):
        return []
    out: List[dict] = []
    for row in rows:
        t_f = safe_pos_float(extract_ts(row), default=None)
        p_f = safe_pos_float(extract_px(row), default=None)
        if t_f is None or p_f is None or (cutoff_s is not None and t_f < cutoff_s):
            continue
        scaled_ts = safe_pos_float(t_f * 1000.0 if ts_in_seconds else t_f, default=None)
        if scaled_ts is None:
            continue
        ts_ms = safe_pos_int(round(scaled_ts), default=None)
        if ts_ms is None:
            continue
        out.append({"timestamp": ts_ms, "price": p_f})
    out.sort(key=lambda x: x["timestamp"])
    return out


def parse_ohlc_interval_minutes(
    interval: Any,
    *,
    str_to_min: Optional[dict[str, int]] = None,
    reject_float: bool = False,
    allow_unit_multiplier: bool = False,
    require_unit_suffix: bool = False,
    venue_label: str = "OHLC",
) -> int:
    """Parse an OHLC interval (integer minutes or string e.g. '1h', '60') into positive integer minutes."""
    if require_unit_suffix and not isinstance(interval, str):
        raise ValueError(f"invalid {venue_label} interval: {interval!r}; examples: 5m, 1h, 1d")
    if isinstance(interval, bool):
        raise ValueError(f"interval must be positive integer or string, not bool: {interval}")
    lookup = str_to_min if str_to_min is not None else OHLC_INTERVALS_STR_TO_MIN
    if isinstance(interval, str):
        cleaned = interval.strip()
        if lookup:
            alias = OHLC_CASE_INSENSITIVE_INTERVAL_ALIASES.get(cleaned.upper(), cleaned)
            if alias not in lookup and alias.lower() in lookup and alias.upper() != "1M":
                alias = alias.lower()
            if alias in lookup:
                return lookup[alias]
        if allow_unit_multiplier and len(cleaned) >= 2:
            num_part, unit = cleaned[:-1], cleaned[-1].lower()
            if num_part.isascii() and num_part.isdigit() and unit in _OHLC_UNIT_MULTIPLIERS:
                minutes = (exact_int(num_part, default=0) or 0) * _OHLC_UNIT_MULTIPLIERS[unit]
                if minutes <= 0:
                    raise ValueError(
                        "the interval must be positive" if require_unit_suffix else f"interval must be positive: {interval}"
                    )
                return minutes
        if require_unit_suffix:
            raise ValueError(f"invalid {venue_label} interval: {interval!r}; examples: 5m, 1h, 1d")
        digits = cleaned[1:] if cleaned.startswith("+") else cleaned
        if digits.isascii() and digits.isdigit():
            val = exact_int(digits, default=0) or 0
            if val <= 0:
                raise ValueError(f"interval must be positive: {interval}")
            return val
        raise ValueError(f"unsupported {venue_label} interval: {interval}")
    if reject_float and isinstance(interval, float):
        raise ValueError(f"invalid interval={interval!r}: unsupported {venue_label} interval")
    val_i = exact_pos_int(interval, default=None)
    if val_i is None:
        raise ValueError(f"invalid interval={interval!r}: unsupported {venue_label} interval")
    return val_i


def parse_ohlc_interval(
    interval: Any,
    *,
    min_to_str: Optional[dict[int, str]] = None,
    str_to_min: Optional[dict[str, int]] = None,
    reject_float: bool = False,
    venue_label: str = "OHLC",
) -> tuple[int, str]:
    """Resolve ``interval`` into ``(interval_minutes, interval_str)`` using ``min_to_str``."""
    m2s = min_to_str if min_to_str is not None else OHLC_INTERVALS_MIN_TO_STR
    s2m = str_to_min if str_to_min is not None else {v: k for k, v in m2s.items()}
    minutes = parse_ohlc_interval_minutes(
        interval, str_to_min=s2m, reject_float=reject_float, venue_label=venue_label
    )
    interval_str = m2s.get(minutes)
    if interval_str is None:
        raise ValueError(f"unsupported {venue_label} interval cadence: {interval} ({minutes}m)")
    return minutes, interval_str


def resolve_ohlc_time_window_ms(
    *,
    start_ms: Any = None,
    end_ms: Any = None,
    lookback_days: Any = None,
    lookback_hours: Any = None,
    timeout: Any = 25,
    default_lookback_days: int = 30,
    strict_numeric_types: bool = False,
) -> tuple[int, int, int, float]:
    """Validate and resolve ``(now_ms, final_start, final_end, timeout_f)`` for OHLC candle fetches."""
    def _valid_num(val: Any, name: str, *, pos: bool = True, scale_ms: float = 1.0) -> float:
        f = (safe_pos_float if pos else safe_nonneg_float)(val, default=None)
        scaled_i = (
            (safe_pos_int if pos else safe_nonneg_int)(f * scale_ms, default=None)
            if f is not None and scale_ms >= 1.0 and name != "timeout"
            else 1
        )
        if f is None or scaled_i is None or (
            strict_numeric_types and not isinstance(val, (int, float))
        ):
            kind = "positive" if pos else "non-negative"
            raise ValueError(f"invalid OHLC window parameters: {name} must be finite and {kind}: {val!r}")
        return f

    now_ms = safe_pos_int(time.time() * 1000.0, default=1) or 1
    timeout_f = _valid_num(timeout, "timeout")
    final_end = safe_pos_int(_valid_num(end_ms, "end_ms"), default=0) if end_ms is not None else now_ms
    if final_end <= 0:
        raise ValueError(f"end_ms must be positive: {end_ms!r}")
    if start_ms is not None:
        final_start = safe_nonneg_int(_valid_num(start_ms, "start_ms", pos=False), default=0) or 0
    elif lookback_days is not None:
        span_ms = safe_pos_int(_valid_num(lookback_days, "lookback_days", scale_ms=86400 * 1000) * 86400 * 1000, default=0) or 0
        final_start = max(0, final_end - span_ms)
    elif lookback_hours is not None:
        span_ms = safe_pos_int(_valid_num(lookback_hours, "lookback_hours", scale_ms=3600 * 1000) * 3600 * 1000, default=0) or 0
        final_start = max(0, final_end - span_ms)
    else:
        final_start = max(0, final_end - default_lookback_days * 86400 * 1000)
    if final_start >= final_end:
        raise ValueError(f"start_ms ({final_start}) must be before end_ms ({final_end})")
    return now_ms, final_start, final_end, timeout_f


_MISSING = object()


def validate_ohlc_candle_bar(
    open_ts: Any,
    open_px: Any,
    high_px: Any,
    low_px: Any,
    close_px: Any,
    *,
    close_ts: Any = _MISSING,
    now_ms: Optional[int] = None,
    drop_incomplete: bool = True,
    close_inclusive: bool = False,
    ts_in_ms: bool = True,
) -> Optional[dict]:
    """Validate and normalize a single OHLC candle into ``{'timestamp': int, 'open': float, 'high': float, 'low': float, 'close': float}``."""
    ot_f = safe_pos_float(open_ts, default=None)
    o_v = safe_pos_float(open_px, default=None)
    h_v = safe_pos_float(high_px, default=None)
    l_v = safe_pos_float(low_px, default=None)
    c_v = safe_pos_float(close_px, default=None)
    if ot_f is None or o_v is None or h_v is None or l_v is None or c_v is None:
        return None
    open_i = math.trunc(ot_f)
    if open_i <= 0:
        return None
    if close_ts is not _MISSING:
        ct_f = safe_pos_float(close_ts, default=None)
        if ct_f is None:
            return None
        close_i = math.trunc(ct_f)
        if close_i < open_i:
            return None
        if (
            _parse_bool_flag(drop_incomplete, True)
            and now_ms is not None
            and ((close_i >= now_ms) if _parse_bool_flag(close_inclusive, False) else (close_i > now_ms))
        ):
            return None
    if h_v < l_v or h_v < max(o_v, c_v) or l_v > min(o_v, c_v):
        return None
    ts_sec = open_i // 1000 if _parse_bool_flag(ts_in_ms, True) else open_i
    if ts_sec <= 0:
        return None
    return {"timestamp": ts_sec, "open": o_v, "high": h_v, "low": l_v, "close": c_v}


REQUIRED_FIELDS = ("timestamp", "open", "high", "low", "close")


def normalize_record(record: Mapping) -> dict:
    """Normalise a row into the common OHLC contract."""
    if not isinstance(record, Mapping):
        raise ValueError("record must be a mapping")
    for field in REQUIRED_FIELDS:
        if field not in record:
            raise ValueError(f"missing required field: {field}")
        if record[field] is None or isinstance(record[field], bool):
            raise ValueError(f"{field} cannot be None or boolean")
    raw_ts = record["timestamp"]
    timestamp = exact_int(raw_ts, default=None)
    if timestamp is None:
        raise ValueError(f"invalid timestamp: {raw_ts!r}")
    try:
        open_f = require_finite_float(record["open"], name="open")
        high_f = require_finite_float(record["high"], name="high")
        low_f = require_finite_float(record["low"], name="low")
        close_f = require_finite_float(record["close"], name="close")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid OHLC numeric value in record: {record!r}") from exc
    return {
        "timestamp": timestamp,
        "open": open_f,
        "high": high_f,
        "low": low_f,
        "close": close_f,
    }


def validate_dataset(records: Iterable[Mapping], *, interval_minutes: int | None = None) -> list[dict]:
    """Validate ordering, cadence and OHLC invariants; return the normalised form."""
    rows = [normalize_record(record) for record in records]
    if not rows:
        raise ValueError("OHLC dataset cannot be empty")
    interval_min = (
        require_int(
            interval_minutes,
            name="interval_minutes",
            min_value=1,
            error_msg="interval_minutes must be positive",
        )
        if interval_minutes is not None
        else None
    )

    expected_delta = interval_min * 60 if interval_min is not None else None
    previous_timestamp = None
    for index, row in enumerate(rows, start=2):
        timestamp = row["timestamp"]
        if previous_timestamp is not None:
            delta = timestamp - previous_timestamp
            if delta <= 0:
                raise ValueError(f"unordered or duplicate timestamp at line {index}: {timestamp}")
            if expected_delta is not None and delta != expected_delta:
                raise ValueError(
                    f"invalid cadence at line {index}: {delta}s, expected {expected_delta}s"
                )
        previous_timestamp = timestamp

        open_, high, low, close = row["open"], row["high"], row["low"], row["close"]
        if not all(safe_pos_float(v, default=None) is not None for v in (open_, high, low, close)):
            raise ValueError(f"non-positive price at line {index}")
        if high < max(open_, close, low) or low > min(open_, close, high):
            raise ValueError(f"invalid OHLC invariant at line {index}")
    return rows


def drop_incomplete_last_bar(
    records: Iterable[Mapping],
    *,
    interval_minutes: int,
    now_timestamp: int,
    regular_session_start: int | None = None,
    regular_session_end: int | None = None,
) -> list[dict]:
    """Drop only the tail of Yahoo candles that are still forming.

    For intraday the timestamp is the start of the bar. For daily bars Yahoo
    uses the session start, so during the session we compare against the
    ``regular`` end provided in the response metadata.
    """
    interval_i = require_int(
        interval_minutes,
        name="interval_minutes",
        min_value=1,
        error_msg="interval_minutes must be positive",
    )
    now_i = require_int(
        now_timestamp,
        name="now_timestamp",
        error_msg="now_timestamp must be an integer",
    )
    if isinstance(regular_session_start, bool) or isinstance(regular_session_end, bool):
        raise ValueError("regular session bounds cannot be boolean")
    sess_start_i = (
        require_int(
            regular_session_start,
            name="regular_session_start",
            error_msg="regular session bounds must be integers",
        )
        if regular_session_start is not None
        else None
    )
    sess_end_i = (
        require_int(
            regular_session_end,
            name="regular_session_end",
            error_msg="regular session bounds must be integers",
        )
        if regular_session_end is not None
        else None
    )
    rows = [normalize_record(record) for record in records]
    if not rows:
        return rows
    if interval_i >= 1440:
        while rows:
            last_timestamp = rows[-1]["timestamp"]
            in_open_session = (
                sess_start_i is not None
                and sess_end_i is not None
                and sess_start_i <= last_timestamp < sess_end_i
                and now_i < sess_end_i
            )
            if last_timestamp > now_i or in_open_session:
                rows.pop()
            else:
                break
        return rows
    interval_seconds = interval_i * 60
    while rows and rows[-1]["timestamp"] + interval_seconds > now_i:
        rows.pop()
    return rows


def validate_ohlc_bars(
    ohlc: Any,
    *,
    exact_len_4: bool = False,
    check_envelope: bool = True,
) -> None:
    """Validate a non-empty sequence of OHLC bars."""
    if not isinstance(ohlc, (list, tuple)) or not ohlc:
        raise ValueError("ohlc cannot be empty")
    for idx, row in enumerate(ohlc):
        if not isinstance(row, (list, tuple)) or (len(row) != 4 if exact_len_4 else len(row) < 4):
            raise ValueError(f"invalid OHLC bar at index {idx}")
        if any(isinstance(row[i], bool) for i in range(4)):
            raise ValueError(f"OHLC bar at index {idx} contains bool")
        vals = [safe_pos_float(row[i], default=None) for i in range(4)]
        if any(v is None for v in vals):
            if any(not isinstance(row[i], (int, float)) and safe_float(row[i], default=None) is None for i in range(4)):
                raise ValueError(f"OHLC bar at index {idx} has non-numeric prices")
            raise ValueError(f"OHLC bar at index {idx} has non-finite or invalid prices")
        o_f, h_f, l_f, c_f = vals[0], vals[1], vals[2], vals[3]
        if (
            h_f < l_f
            or (check_envelope and (h_f < max(o_f, c_f) or l_f > min(o_f, c_f)))
        ):
            raise ValueError(f"OHLC bar at index {idx} has non-finite or invalid prices")


def align_previous_values(
    target_timestamps: Sequence[int],
    source_records: Iterable[Mapping],
    *,
    value_field: str = "close",
    transform: Optional[Callable[[float], float]] = None,
) -> List[float]:
    """Align as-of without lookahead: the last value with timestamp <= target."""
    by_ts: dict[int, float] = {}
    for record in source_records:
        raw_ts = record["timestamp"]
        raw_val = record[value_field]
        if isinstance(raw_ts, bool) or isinstance(raw_val, bool) or raw_ts is None or raw_val is None:
            raise ValueError("source_records cannot contain None or boolean timestamp/value")
        ts_i = exact_int(raw_ts, default=None)
        val_f = safe_float(raw_val, default=None)
        if ts_i is None or val_f is None:
            raise ValueError("invalid or non-finite timestamp or value in source_records")
        by_ts[ts_i] = val_f
    source = sorted(by_ts.items(), key=lambda item: item[0])
    if not source:
        raise ValueError("the source series for alignment cannot be empty")
    convert = transform or (lambda value: value)
    result: List[float] = []
    source_index = 0
    current: Optional[float] = None
    prev_target: Optional[int] = None
    for timestamp in target_timestamps:
        if isinstance(timestamp, bool) or timestamp is None:
            raise ValueError("target_timestamps cannot contain None or boolean")
        target = exact_int(timestamp, default=None)
        if target is None:
            raise ValueError(f"invalid or non-finite target timestamp: {timestamp!r}")
        if prev_target is not None and target < prev_target:
            raise ValueError(f"target_timestamps must be sorted in non-decreasing order at {target}")
        prev_target = target
        while source_index < len(source) and source[source_index][0] <= target:
            current = convert(source[source_index][1])
            source_index += 1
        if current is None:
            raise ValueError(
                f"no known FX value at or before timestamp {target}; "
                "extend the FX dataset further back"
            )
        cur_f = safe_pos_float(current, default=None)
        if cur_f is None:
            raise ValueError(f"non-positive aligned value at timestamp {target}")
        result.append(cur_f)
    return result


def candles_to_historical_points(candles: Any) -> list[tuple[float, float]]:
    """Extract ``(epoch_sec, close_price)`` tuples from OHLC candle dictionaries."""
    if not isinstance(candles, Iterable) or isinstance(candles, (str, bytes, Mapping)):
        return []
    out: list[tuple[float, float]] = []
    for c in candles:
        if not isinstance(c, Mapping):
            continue
        tf = normalize_epoch_sec(c.get("timestamp"), default=None)
        pf = safe_pos_float(c.get("close"), default=None)
        if tf is not None and pf is not None:
            out.append((tf, pf))
    return out


def series_to_historical_points(series: Any) -> list[tuple[float, float]]:
    """Extract ``(epoch_sec, close_price)`` tuples from a ``ClosedPriceSeries``-like object."""
    raw_ts = tuple(getattr(series, "timestamps", ()) or ())
    raw_px = tuple(getattr(series, "closes", ()) or ())
    if len(raw_ts) != len(raw_px):
        return []
    return [
        (tv, pv)
        for t, p in zip(raw_ts, raw_px)
        if (tv := safe_pos_float(t, default=None)) is not None
        and (pv := safe_pos_float(p, default=None)) is not None
    ]


def extract_trading_quote_pairs(
    data: Any, *, quote: str = "USDC", strict: bool = False
) -> tuple[set[str], set[str], dict[str, str]]:
    """Extract ``(trading_symbols, quote_pairs, pair_to_base)`` from an ``exchangeInfo`` payload."""
    raw_symbols = data.get("symbols") if isinstance(data, Mapping) else None
    if not isinstance(raw_symbols, list):
        if strict:
            raise ValueError(
                "exchangeInfo response must be a dict"
                if not isinstance(data, Mapping)
                else "exchangeInfo symbols must be a list"
            )
        return set(), set(), {}
    want_quote = quote.strip().upper()
    trading_symbols: set[str] = set()
    quote_pairs: set[str] = set()
    pair_to_base: dict[str, str] = {}
    for info in raw_symbols:
        if not isinstance(info, Mapping):
            continue
        raw_sym = info.get("symbol")
        raw_base = info.get("baseAsset")
        raw_quote = info.get("quoteAsset")
        raw_status = info.get("status")
        sym = raw_sym.strip().upper() if isinstance(raw_sym, str) else ""
        base = raw_base.strip().upper() if isinstance(raw_base, str) else ""
        q_asset = raw_quote.strip().upper() if isinstance(raw_quote, str) else ""
        status = raw_status.strip().upper() if isinstance(raw_status, str) else ""
        if not sym or status != "TRADING":
            continue
        trading_symbols.add(sym)
        if q_asset == want_quote:
            quote_pairs.add(sym)
            if base:
                pair_to_base[sym] = base
    return trading_symbols, quote_pairs, pair_to_base


def extract_yahoo_chart_result_and_quote(payload: Any) -> tuple[Optional[dict], dict, list, dict]:
    """Unpack a Yahoo v8 chart JSON payload into ``(result, chart_dict, timestamps, quote_dict)``."""
    chart = payload.get("chart") if isinstance(payload, Mapping) else None
    chart_d = dict(chart) if isinstance(chart, Mapping) else {}
    raw_results = chart_d.get("result")
    result = (
        raw_results[0]
        if isinstance(raw_results, Sequence) and not isinstance(raw_results, (str, bytes)) and raw_results
        else None
    )
    if not isinstance(result, dict) or not result:
        return None, chart_d, [], {}
    raw_ts = result.get("timestamp")
    timestamps = raw_ts if isinstance(raw_ts, list) else []
    indicators = result.get("indicators") if isinstance(result.get("indicators"), Mapping) else {}
    quote_list = indicators.get("quote") if isinstance(indicators.get("quote"), list) else []
    quote = (quote_list[0] if quote_list and isinstance(quote_list[0], dict) else {}) or {}
    return result, chart_d, timestamps, quote


def extract_yahoo_chart_latest_price(payload: Any, *, decimals: Optional[int] = None) -> Optional[float]:
    """Extract the latest positive intraday close (or ``meta.regularMarketPrice`` fallback) from a Yahoo chart payload."""
    result, _, _, quote = extract_yahoo_chart_result_and_quote(payload)
    if result is None:
        return None
    raw_closes = quote.get("close")
    if isinstance(raw_closes, Sequence) and not isinstance(raw_closes, (str, bytes)):
        for c in reversed(raw_closes):
            if (cf := safe_pos_float(c, default=None)) is not None:
                return round(cf, decimals) if decimals is not None else cf
    meta = result.get("meta")
    val = safe_pos_float(meta.get("regularMarketPrice") if isinstance(meta, Mapping) else None, default=None)
    if val is None:
        return None
    return round(val, decimals) if decimals is not None else val


def normalize_yahoo_5d_chart_bars(
    chart_res: Any,
    raw_series_24h: list,
    now_ts: float,
    *,
    eq_rth_open: bool = False,
) -> tuple[list[list[Any]], list[list[Any]], list[list[Any]], bool, bool, float, int]:
    """Parse Yahoo 5d/5m chart payload, prune off-RTH zero-vol tail gaps, slice 24h window, and stitch sub-5m ticks."""
    pts_5d: list[list[Any]] = []
    if isinstance(chart_res, tuple) and len(chart_res) >= 2 and isinstance(chart_res[1], list):
        pts_5d = sorted(
            (
                (
                    [math.trunc(ts_f if ts_f > 1e11 else ts_f * 1000), round(px_f, 4), safe_nonneg_float(bar[2], 0.0) or 0.0]
                    if len(bar) >= 3
                    else [math.trunc(ts_f if ts_f > 1e11 else ts_f * 1000), round(px_f, 4)]
                )
                for bar in chart_res[1]
                if isinstance(bar, (list, tuple))
                and len(bar) >= 2
                and (ts_f := safe_pos_float(bar[0], default=None)) is not None
                and (px_f := safe_pos_float(bar[1], default=None)) is not None
            ),
            key=lambda pt: pt[0],
        )
    elif isinstance(chart_res, dict):
        q_obj = ((chart_res.get("indicators") or {}).get("quote") or [{}])[0]
        vols = q_obj.get("volume") or []
        pts_5d = sorted(
            (
                [
                    math.trunc(ts_f if ts_f > 1e11 else ts_f * 1000),
                    round(c_f, 4),
                    safe_nonneg_float(vols[idx], 0.0) or 0.0 if idx < len(vols) else 0.0,
                ]
                for idx, (ts, c) in enumerate(zip(chart_res.get("timestamp") or [], q_obj.get("close") or []))
                if (ts_f := safe_pos_float(ts, default=None)) is not None and (c_f := safe_pos_float(c, default=None)) is not None
            ),
            key=lambda pt: pt[0],
        )
    if not pts_5d:
        return [], [], [], False, False, 0.0, 0
    chart_px = safe_pos_float(pts_5d[-1][1], default=0.0) or 0.0
    now_ms_i = math.trunc(now_ts * 1000)
    raw_last_5d_ts = math.trunc(pts_5d[-1][0])
    if len(pts_5d) >= 10:
        for split_idx in range(len(pts_5d) - 1, max(4, len(pts_5d) - 4), -1):
            if (
                (pts_5d[split_idx][0] - pts_5d[split_idx - 1][0]) > 2 * 3600 * 1000
                and (pts_5d[-1][0] - pts_5d[split_idx][0]) < 15 * 60 * 1000
            ):
                pts_5d = pts_5d[:split_idx]
                break
        last_pos_v_idx = max(
            (
                idx_v
                for idx_v, pt_v in enumerate(pts_5d)
                if len(pt_v) >= 3 and (safe_pos_float(pt_v[2], default=0.0) or 0.0) > 0.0
            ),
            default=-1,
        )
        if 4 <= last_pos_v_idx < len(pts_5d) - 1:
            while len(pts_5d) - 1 > last_pos_v_idx:
                tail_gap_idx = next(
                    (
                        zv_idx
                        for zv_idx in range(len(pts_5d) - 1, last_pos_v_idx, -1)
                        if (pts_5d[zv_idx][0] - pts_5d[zv_idx - 1][0]) > 20 * 60 * 1000
                    ),
                    None,
                )
                if tail_gap_idx is None:
                    break
                tail_span_ms = pts_5d[-1][0] - pts_5d[tail_gap_idx][0]
                tail_is_live = (now_ms_i - pts_5d[-1][0]) <= 20 * 60 * 1000
                if tail_span_ms < 15 * 60 * 1000 and not tail_is_live:
                    pts_5d = pts_5d[:tail_gap_idx]
                else:
                    break
    rth_bar_fresh = (now_ms_i - pts_5d[-1][0]) <= (4 * 3600 * 1000 if eq_rth_open else 20 * 60 * 1000)
    chart_tick_fresh = (now_ms_i - raw_last_5d_ts) <= 20 * 60 * 1000
    anchor_24h_ms = min(now_ms_i, math.trunc(pts_5d[-1][0])) if pts_5d else now_ms_i
    cutoff_24h_ms = anchor_24h_ms - 86400 * 1000
    pts_24h_bars = [pt for pt in pts_5d if pt[0] >= cutoff_24h_ms]
    if pts_24h_bars and (pts_24h_bars[-1][0] - pts_24h_bars[0][0]) < 18 * 3600 * 1000:
        older_bars = [pt for pt in pts_5d if pt[0] < cutoff_24h_ms]
        if older_bars:
            pts_24h_bars = [older_bars[-1], *pts_24h_bars]
    if len(pts_24h_bars) < 10:
        pts_24h_bars = list(pts_5d[-78:])
    pts_24h = list(pts_24h_bars)
    if rth_bar_fresh and raw_series_24h and pts_24h:
        last_bar_ts = pts_24h[-1][0]
        recent_bar_pxs = [
            b[1]
            for b in pts_24h_bars
            if isinstance(b, (list, tuple)) and len(b) >= 2 and b[0] >= last_bar_ts - 15 * 60 * 1000
        ]
        if len(pts_24h_bars) >= 2 and isinstance(pts_24h_bars[-2], (list, tuple)) and len(pts_24h_bars[-2]) >= 2:
            recent_bar_pxs.append(pts_24h_bars[-2][1])
        bar_win_lo = min(recent_bar_pxs) if recent_bar_pxs else pts_24h[-1][1]
        bar_win_hi = max(recent_bar_pxs) if recent_bar_pxs else pts_24h[-1][1]
        upper_sub5m_ms = (now_ms_i + 5000) if eq_rth_open else min(now_ms_i + 5000, last_bar_ts + 300 * 1000)
        recent_sub5m = [
            pt
            for pt in raw_series_24h
            if isinstance(pt, (list, tuple))
            and len(pt) >= 2
            and last_bar_ts - 15 * 60 * 1000 <= pt[0] <= upper_sub5m_ms
            and not _yahoo_bar_prices_diverged(pt[1], pts_24h[-1][1])
            and (pt[0] >= last_bar_ts or (bar_win_lo * 0.995 <= pt[1] <= bar_win_hi * 1.005))
        ]
        if recent_sub5m:
            by_ts = {pt[0]: pt for pt in pts_24h}
            for r_pt in recent_sub5m:
                by_ts.setdefault(r_pt[0], r_pt)
            pts_24h = [by_ts[k] for k in sorted(by_ts)]
    return pts_5d, pts_24h_bars, pts_24h, rth_bar_fresh, chart_tick_fresh, chart_px, raw_last_5d_ts
