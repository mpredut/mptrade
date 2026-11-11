#!/usr/bin/env python3
"""
market_data.py -- Yahoo Finance market data: prices, FX rates, and real-trading detection.
"""

from __future__ import annotations

import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from botcore import (
    clean_nonempty_str,
    exact_pos_int,
    fetch_public_json,
    http_get,
    log,
    normalize_epoch_sec,
    safe_float,
    safe_int,
    safe_nonneg_float,
    safe_pos_float,
)

from providers.order_ohlc_utils import (
    drop_incomplete_last_bar,
    extract_yahoo_chart_result_and_quote as _extract_yahoo_result_and_quote,
    parse_ohlc_interval_minutes,
    validate_dataset,
    validate_ohlc_candle_bar,
)
from state_io import parse_json_payload


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
    if (sym_clean := clean_nonempty_str(symbol)) is None:
        raise ValueError(f"invalid symbol: {symbol!r}")
    if (rng_clean := clean_nonempty_str(range_)) is None:
        raise ValueError(f"invalid range: {range_!r}")
    if (itv_clean := clean_nonempty_str(interval)) is None:
        raise ValueError(f"invalid interval: {interval!r}")
    symbol = sym_clean
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
    if drop_incomplete:
        regular = ((result.get("meta") or {}).get("currentTradingPeriod") or {}).get(
            "regular"
        ) or {}
        records = drop_incomplete_last_bar(
            records,
            interval_minutes=parse_interval_minutes(interval),
            now_timestamp=int(time.time()),
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
        result, _, ts, q = _extract_yahoo_result_and_quote(parse_json_payload(body))
        if result is None:
            return None, []
        meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
        closes = q.get("close") if isinstance(q.get("close"), list) else []
        vols = q.get("volume") if isinstance(q.get("volume"), list) else []
        bars = []
        for i, (t, c) in enumerate(zip(ts, closes)):
            t_f = safe_pos_float(t, default=None)
            c_f = safe_pos_float(c, default=None)
            if t_f is None or c_f is None:
                continue
            raw_v = vols[i] if i < len(vols) else None
            v_clean = safe_nonneg_float(raw_v, default=None)
            bars.append((int(t_f) if t_f.is_integer() else t_f, c_f, v_clean))
        reg_period = (
            ((meta.get("currentTradingPeriod") or {}).get("regular") or {})
            if isinstance(meta.get("currentTradingPeriod"), dict)
            else {}
        )
        reg_start = safe_pos_float(reg_period.get("start"), default=None) if isinstance(reg_period, dict) else None
        reg_end = safe_pos_float(reg_period.get("end"), default=None) if isinstance(reg_period, dict) else None
        ref_meta_px = (
            safe_pos_float(meta.get("regularMarketPrice"), default=None)
            or safe_pos_float(meta.get("chartPreviousClose"), default=None)
        )
        if reg_start is not None and bars:
            reg_bars = [b for b in bars if reg_start <= b[0] <= (reg_end or float("inf"))]
            cleaned_pre_post = []
            for idx, b in enumerate(bars):
                is_outside_reg = b[0] < reg_start or (reg_end is not None and b[0] > reg_end)
                if is_outside_reg and (b[2] is None or b[2] <= 0.0):
                    ref_px = (
                        (reg_bars[0][1] if b[0] < reg_start else reg_bars[-1][1])
                        if reg_bars
                        else ref_meta_px
                    )
                    if ref_px is not None and ref_px > 0.0 and abs(b[1] - ref_px) / ref_px > 0.01:
                        near_bars = [
                            nb for j, nb in enumerate(bars)
                            if j != idx and abs(nb[0] - b[0]) <= 1800 and b[1] > 0.0 and abs(nb[1] - b[1]) / b[1] <= 0.01
                        ]
                        has_session_peer = len(near_bars) >= 2 or any(abs(nb[1] - b[1]) > 1e-9 for nb in near_bars)
                        if not has_session_peer:
                            continue
                cleaned_pre_post.append(b)
            if cleaned_pre_post:
                bars = cleaned_pre_post
        if len(bars) >= 3 and any((b[2] or 0.0) > 0.0 for b in bars):
            filtered_bars = [bars[0]]
            for idx in range(1, len(bars) - 1):
                cur_b = bars[idx]
                prev_b = filtered_bars[-1]
                cur_v = cur_b[2]
                if cur_v is None or cur_v <= 0.0:
                    next_idx = idx + 1
                    while (
                        next_idx < len(bars) - 1
                        and (bars[next_idx][2] is None or bars[next_idx][2] <= 0.0)
                        and abs(bars[next_idx][1] - cur_b[1]) <= 1e-9
                    ):
                        next_idx += 1
                    next_b = bars[next_idx]
                    if (prev_b[2] or 0.0) > 0.0 or (next_b[2] or 0.0) > 0.0:
                        p_prev, p_cur, p_next = prev_b[1], cur_b[1], next_b[1]
                        ref_mid = 0.5 * (p_prev + p_next)
                        if (
                            ref_mid > 0.0
                            and abs(p_prev - p_next) / ref_mid <= 0.025
                            and abs(p_cur - ref_mid) / ref_mid > 0.025
                        ):
                            continue
                filtered_bars.append(cur_b)
            filtered_bars.append(bars[-1])
            bars = filtered_bars
        return meta, bars
    except (ValueError, KeyError, TypeError):
        return None, []


def get_price_usd(sym: str) -> float | None:
    """Return current price for NVDA, SPCX, USDRON=X, etc.
    Prefer the fresher final series bar and use regularMarketPrice only if absent."""
    meta, bars = _chart(sym)
    if bars:
        return bars[-1][1]
    if meta:
        return safe_pos_float(meta.get("regularMarketPrice"), default=None)
    return None


def compute_ols_slope_pct(closes, bars: int = 12) -> float | None:
    """Compute short-term OLS trend slope percentage per bar over the last ``bars`` closes."""
    if closes is None or isinstance(closes, bool):
        return None
    bars_i = safe_int(bars, default=None)
    bars_n = max(4, bars_i) if bars_i is not None else 12
    try:
        clean_closes = [
            cf for c in closes
            if (cf := safe_pos_float(c, default=None)) is not None
        ][-bars_n:]
    except TypeError:
        return None
    if len(clean_closes) < max(4, bars_n // 2):
        return None
    n = len(clean_closes)
    mx = (n - 1) / 2.0
    my = statistics.fmean(clean_closes)
    denom = n * (n * n - 1) / 12.0
    my_f = safe_pos_float(my, default=None)
    if denom == 0 or my_f is None:
        return None
    slope = sum((i - mx) * (clean_closes[i] - my_f) for i in range(n)) / denom
    return safe_float(slope / my_f * 100.0, default=None)


def trend_slope_pct(sym: str, bars: int = 12) -> float | None:
    """Return short-term OLS trend slope over the last `bars` five-minute closes.
    Normalize as price percentage PER BAR; negative means downtrend. Return None with
    insufficient data. Used as a DCA gate like Binance/Kraken to avoid falling markets."""
    _, bars_rows = _chart(sym)
    return compute_ols_slope_pct([c for (_, c, _) in bars_rows], bars=bars)


def _fx_rate(pair: str, label: str, min_rate: float, fallback: float) -> float:
    rate = safe_pos_float(get_price_usd(pair), default=None)
    if rate is not None and rate > min_rate:
        return rate
    log(f"  ! {label} rate unavailable, using fallback {fallback}")
    return fallback


def get_usd_ron() -> float:
    """Return current USD/RON, falling back to 4.65 when the feed is unavailable."""
    return _fx_rate("USDRON=X", "USD/RON", 1.0, 4.65)


def get_eur_usd() -> float:
    """Return current EUR->USD (USD per EUR), falling back to 1.08."""
    return _fx_rate("EURUSD=X", "EUR/USD", 0.5, 1.08)


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
