#!/usr/bin/env python3
"""
market_data.py -- Yahoo Finance market data: prices, FX rates, and real-trading detection.
"""

from __future__ import annotations

from functools import partial
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from botcore import (
    SHORT_TREND_FRESHNESS_MAX_AGE_SEC,
    _parse_bool_flag,
    clean_nonempty_str,
    exact_pos_int,
    fetch_public_json,
    http_get,
    is_shared_disk_cache_enabled,
    log,
    normalize_epoch_sec,
    safe_int,
    safe_nonneg_float,
    safe_pos_float,
)
from cachepaths import (
    CACHE_CURRENT_PRICE_FILENAME,
    ensure_equity_suffix,
    resolve_cachedb_filepath,
)
from providers.ohlc_utils import (
    compute_ols_slope_pct,
    drop_incomplete_last_bar,
    extract_yahoo_chart_result_and_quote as _extract_yahoo_result_and_quote,
    extract_yahoo_intraday_chart_bars,
    parse_ohlc_interval_minutes,
    require_ohlc_symbol,
    validate_dataset,
    validate_ohlc_candle_bar,
)
from providers.ssot_bridge import ProviderSsotBridgeMixin, is_in_live_price_fetch
from state_io import parse_json_payload

_ORIG_HTTP_GET = http_get
_SSOT_CACHE_DIR_OVERRIDE: str | None = None


def parse_interval_minutes(value: str) -> int:
    """Parse interval strings like '5m', '1h', '1d' into integer minutes."""
    return parse_ohlc_interval_minutes(
        value,
        str_to_min={},
        allow_unit_multiplier=True,
        require_unit_suffix=True,
        venue_label="Yahoo",
    )


def fetch_yahoo_candles(
    symbol: str,
    range_: str,
    interval: str,
    *,
    drop_incomplete: bool = True,
    timeout: int = 25,
) -> list[dict]:
    """Fetch completed Yahoo Finance OHLC bars as canonical records.

    Excludes the forming candle when drop_incomplete is True to prevent lookahead drift.
    """
    to_f = safe_pos_float(timeout, default=None)
    if to_f is None:
        raise ValueError(f"invalid timeout: {timeout!r}")
    symbol = require_ohlc_symbol(symbol, label="symbol")
    if (rng_clean := clean_nonempty_str(range_)) is None:
        raise ValueError(f"invalid range: {range_!r}")
    if (itv_clean := clean_nonempty_str(interval)) is None:
        raise ValueError(f"invalid interval: {interval!r}")
    range_ = rng_clean
    interval = itv_clean
    query = {"range": range_, "interval": interval}
    day_range = re.fullmatch(r"([1-9][0-9]*)d", range_.lower())
    if day_range:
        end = safe_int(time.time(), 0)
        days = exact_pos_int(day_range.group(1), default=1) or 1
        span_sec = max(60, days * 24 * 60 * 60 - 60)
        query = {
            "period1": end - span_sec,
            "period2": end,
            "interval": interval,
        }
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{urllib.parse.quote(symbol)}?{urllib.parse.urlencode(query)}"
    )
    try:
        payload = fetch_public_json(url, timeout=to_f, user_agent="mptrade/t212-yahoo-fetcher")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(
            f"Yahoo refused {symbol} {range_}/{interval}: HTTP {exc.code} {detail}"
        ) from exc
    result, chart, timestamps, quote = _extract_yahoo_result_and_quote(payload)
    if result is None:
        error = chart.get("error")
        raise RuntimeError(f"Yahoo returned no data for {symbol}: {error}")
    opens = quote.get("open") if isinstance(quote.get("open"), list) else []
    highs = quote.get("high") if isinstance(quote.get("high"), list) else []
    lows = quote.get("low") if isinstance(quote.get("low"), list) else []
    closes = quote.get("close") if isinstance(quote.get("close"), list) else []
    series_len = min(len(opens), len(highs), len(lows), len(closes))
    records = []
    for index, timestamp in enumerate(timestamps[:series_len]):
        bar = validate_ohlc_candle_bar(
            timestamp, opens[index], highs[index], lows[index], closes[index],
            drop_incomplete=False, ts_in_ms=False,
        )
        if bar is not None:
            records.append(bar)
    if _parse_bool_flag(drop_incomplete, True):
        regular = ((result.get("meta") or {}).get("currentTradingPeriod") or {}).get(
            "regular"
        ) or {}
        records = drop_incomplete_last_bar(
            records,
            interval_minutes=parse_interval_minutes(interval),
            now_timestamp=math.trunc(time.time()),
            regular_session_start=regular.get("start"),
            regular_session_end=regular.get("end"),
        )
    return validate_dataset(records)

YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
_UA = {"User-Agent": "Mozilla/5.0 (ipo-watch)"}


def _chart(sym: str, rng: str = "1d", interval: str = "5m"):
    """Return (meta, bars) from Yahoo chart data; bars contain non-null closes.
    The intraday series is fresher than metadata. For a new listing, metadata can remain
    stale, e.g. SPCX volume=0 and an old price despite active trading."""
    if isinstance(sym, bool) or not isinstance(sym, str) or not sym.strip():
        return None, []
    sym_clean = sym.strip()
    qs = urllib.parse.urlencode(
        {
            "range": str(rng or "1d").strip(),
            "interval": str(interval or "5m").strip(),
            "includePrePost": "true",
        }
    )
    status, body = http_get(YAHOO_CHART.format(sym=urllib.parse.quote(sym_clean, safe="")) + f"?{qs}",
                            headers=_UA)
    if status != 200 or not body:
        return None, []
    try:
        return extract_yahoo_intraday_chart_bars(parse_json_payload(body))
    except (ValueError, KeyError, TypeError):
        return None, []


_ORIG_CHART = _chart


def _can_use_ssot_equity_cache(sym: str) -> str | None:
    if isinstance(sym, bool) or not isinstance(sym, str) or not (sym_u := sym.strip().upper()):
        return None
    if (
        sym_u.endswith("=X")
        or is_in_live_price_fetch()
        or http_get is not _ORIG_HTTP_GET
        or _chart is not _ORIG_CHART
        or (_SSOT_CACHE_DIR_OVERRIDE is None and not is_shared_disk_cache_enabled())
    ):
        return None
    return sym_u


def _equity_ssot_candidates(sym_u: str, *, include_bare: bool = True) -> tuple[str, ...]:
    raw = (
        ensure_equity_suffix(sym_u),
        ensure_equity_suffix(sym_u, suffix="_EQ"),
        *(
            (sym_u,)
            if include_bare
            else ()
        ),
    )
    return tuple(dict.fromkeys(c for c in raw if c))


def _ssot_equity_price_usd(sym_u: str) -> float | None:
    price_path = (
        str(resolve_cachedb_filepath(os.path.join(_SSOT_CACHE_DIR_OVERRIDE, CACHE_CURRENT_PRICE_FILENAME)))
        if _SSOT_CACHE_DIR_OVERRIDE
        else None
    )
    return safe_pos_float(
        ProviderSsotBridgeMixin.read_ssot_spot_price(
            _equity_ssot_candidates(sym_u, include_bare=True),
            max_age_sec=SHORT_TREND_FRESHNESS_MAX_AGE_SEC,
            price_cache_path=price_path,
        ),
        default=None,
    )


def _ssot_equity_ohlc_closes(sym_u: str) -> list[float]:
    return ProviderSsotBridgeMixin.read_ssot_ohlc_24h_closes(
        _equity_ssot_candidates(sym_u, include_bare=False),
        max_age_sec=parse_interval_minutes("5m") * 60.0,
        cache_dir_override=_SSOT_CACHE_DIR_OVERRIDE,
    )


def get_price_usd(sym: str) -> float | None:
    """Return current price for NVDA, SPCX, USDRON=X, etc.
    Prefer fresh SSOT cacheManager equity quotes when active, else Yahoo chart bars."""
    if (sym_u := _can_use_ssot_equity_cache(sym)) is not None:
        if (ssot_px := _ssot_equity_price_usd(sym_u)) is not None:
            return ssot_px
    meta, bars = _chart(sym)
    if bars:
        return bars[-1][1]
    if meta:
        return safe_pos_float(meta.get("regularMarketPrice"), default=None)
    return None


def trend_slope_pct(sym: str, bars: int = 12) -> float | None:
    """Return short-term OLS trend slope over the last `bars` five-minute closes.
    Normalize as price percentage PER BAR; negative means downtrend. Return None with
    insufficient data. Used as a DCA gate like Binance/Kraken to avoid falling markets."""
    if (sym_u := _can_use_ssot_equity_cache(sym)) is not None:
        if (ssot_slope := compute_ols_slope_pct(_ssot_equity_ohlc_closes(sym_u), bars=bars)) is not None:
            return ssot_slope
    _, bars_rows = _chart(sym)
    return compute_ols_slope_pct([c for (_, c, _) in bars_rows], bars=bars)


def _fx_rate(pair: str, label: str, min_rate: float, fallback: float) -> float:
    rate = safe_pos_float(get_price_usd(pair), default=None)
    if rate is not None and rate > min_rate:
        return rate
    log(f"  ! {label} rate unavailable, using fallback {fallback}")
    return fallback


get_usd_ron = partial(_fx_rate, "USDRON=X", "USD/RON", 1.0, 4.65)
get_eur_usd = partial(_fx_rate, "EURUSD=X", "EUR/USD", 0.5, 1.08)


def check_market(sym: str) -> dict | None:
    """Return metadata with trading=True only when the symbol is actually trading.

    Avoid false positives from fixed-price, zero-volume IPO placeholders by requiring
    positive volume, a recent trade within 15 minutes, and an active market state.
    """
    meta, bars = _chart(sym)
    if meta is None and not bars:
        return None
    meta = meta or {}

    price = safe_pos_float(meta.get("regularMarketPrice"), default=None)
    meta_vol = safe_nonneg_float(meta.get("regularMarketVolume"), default=0)
    raw_state = meta.get("marketState")
    state    = raw_state.strip().upper() if isinstance(raw_state, str) else ""
    now_ts = time.time()
    ts_f = normalize_epoch_sec(meta.get("regularMarketTime"), default=None)
    age_sec = (now_ts - ts_f) if ts_f is not None else None

    # --- intraday SERIES signals robust to stale metadata ---
    series_age = series_vol = None
    series_price = None
    series_moved = False
    if bars:
        recent = bars[-6:]                       # approximately the last 30 minutes
        last_bar_ts = normalize_epoch_sec(bars[-1][0], default=None)
        series_age = (now_ts - last_bar_ts) if last_bar_ts is not None else None
        series_price = safe_pos_float(bars[-1][1], default=None)
        series_vol = sum(safe_nonneg_float(v, default=0.0) for _, _, v in recent)
        cs = [c_f for _, c, _ in recent if (c_f := safe_pos_float(c, default=None)) is not None]
        series_moved = len(cs) >= 2 and (max(cs) - min(cs)) > 0

    fresh_meta = age_sec is not None and -60.0 <= age_sec < 15 * 60
    fresh_series = series_age is not None and -60.0 <= series_age < 20 * 60
    live_state = state in ("REGULAR", "PRE", "PREPRE", "POST", "POSTPOST")

    # 'launched' means actual prior trading: metadata volume OR a fresh intraday series
    # showing recent volume or price movement. This detects SPCX despite stale zero-volume
    # metadata when its series contains live bars, without mistaking a missing or flat
    # pre-IPO placeholder series for a launch.
    launched = (bool(price) and meta_vol > 0) or \
               (fresh_series and ((series_vol or 0) > 0 or series_moved))
    really_trading = launched and (fresh_meta or fresh_series) and (live_state or series_moved)

    age_min = None
    eff_age = series_age if series_age is not None else age_sec
    if eff_age is not None:
        age_min = round(eff_age / 60, 1)

    return {
        "price":    series_price or price,   # prefer the fresher series price
        "currency": meta.get("currency"),
        "exchange": meta.get("exchangeName") or meta.get("fullExchangeName"),
        "volume":   meta_vol or (series_vol or 0),
        "state":    state or "?",
        "age_min":  age_min,
        "name":     meta.get("longName") or meta.get("shortName") or "",
        "trading":  really_trading,   # trading NOW
        "launched": launched,         # has begun trading
    }
