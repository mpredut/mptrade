#!/usr/bin/env python3
"""
market_data.py — Yahoo Finance market data: prices, FX rates, and real-trading detection.
"""

from __future__ import annotations

import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request

try:
    from ipo_common import http_get, log
except ImportError:
    try:
        from .ipo_common import http_get, log
    except ImportError:
        from botcore import http_get, log

from cachepaths import canonical_base_asset
from offline.backtests.datasets import drop_incomplete_last_bar, validate_dataset


def parse_interval_minutes(value: str) -> int:
    """Parse interval strings like '5m', '1h', '1d' into integer minutes."""
    if not isinstance(value, str):
        raise ValueError(f"invalid Yahoo interval: {value!r}; examples: 5m, 1h, 1d")
    cleaned = value.strip()
    if len(cleaned) < 2:
        raise ValueError(f"invalid Yahoo interval: {value!r}; examples: 5m, 1h, 1d")
    units = {"m": 1, "h": 60, "d": 1440}
    try:
        minutes = int(cleaned[:-1]) * units[cleaned[-1].lower()]
    except (KeyError, ValueError, IndexError, TypeError) as exc:
        raise ValueError(f"invalid Yahoo interval: {value!r}; examples: 5m, 1h, 1d") from exc
    if minutes <= 0:
        raise ValueError("the interval must be positive")
    return minutes


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
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(float(timeout)) or float(timeout) <= 0:
        raise ValueError(f"invalid timeout: {timeout!r}")
    if isinstance(symbol, bool) or not isinstance(symbol, str) or not symbol.strip():
        raise ValueError(f"invalid symbol: {symbol!r}")
    if isinstance(range_, bool) or not isinstance(range_, str) or not range_.strip():
        raise ValueError(f"invalid range: {range_!r}")
    if isinstance(interval, bool) or not isinstance(interval, str) or not interval.strip():
        raise ValueError(f"invalid interval: {interval!r}")
    symbol = symbol.strip()
    range_ = range_.strip()
    interval = interval.strip()
    query = {"range": range_, "interval": interval}
    day_range = re.fullmatch(r"([1-9][0-9]*)d", range_.lower())
    if day_range:
        end = int(time.time())
        span_sec = max(60, int(day_range.group(1)) * 24 * 60 * 60 - 60)
        query = {
            "period1": end - span_sec,
            "period2": end,
            "interval": interval,
        }
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{urllib.parse.quote(symbol)}?{urllib.parse.urlencode(query)}"
    )
    request = urllib.request.Request(
        url, headers={"User-Agent": "mptrade/t212-yahoo-fetcher"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(
            f"Yahoo refused {symbol} {range_}/{interval}: HTTP {exc.code} {detail}"
        ) from exc
    chart = payload.get("chart") if isinstance(payload, dict) else None
    chart = chart if isinstance(chart, dict) else {}
    result = (chart.get("result") or [None])[0]
    if not isinstance(result, dict) or not result:
        error = chart.get("error")
        raise RuntimeError(f"Yahoo returned no data for {symbol}: {error}")
    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators") if isinstance(result.get("indicators"), dict) else {}
    quote_list = indicators.get("quote") if isinstance(indicators.get("quote"), list) else []
    quote = (quote_list[0] if quote_list and isinstance(quote_list[0], dict) else {}) or {}
    opens = quote.get("open") or []
    highs = quote.get("high") or []
    lows = quote.get("low") or []
    closes = quote.get("close") or []
    series_len = min(len(opens), len(highs), len(lows), len(closes))
    records = []
    for index, timestamp in enumerate(timestamps):
        if timestamp is None or isinstance(timestamp, bool) or index >= series_len:
            continue
        values = (opens[index], highs[index], lows[index], closes[index])
        if None in values or any(isinstance(v, bool) for v in values):
            continue
        try:
            ts_f = float(timestamp)
            o_f, h_f, l_f, c_f = float(values[0]), float(values[1]), float(values[2]), float(values[3])
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(ts_f):
            continue
        ts_i = int(ts_f)
        if ts_i <= 0 or not all(math.isfinite(v) and v > 0 for v in (o_f, h_f, l_f, c_f)):
            continue
        if h_f < l_f or h_f < max(o_f, c_f) or l_f > min(o_f, c_f):
            continue
        records.append({
            "timestamp": ts_i,
            "open": o_f,
            "high": h_f,
            "low": l_f,
            "close": c_f,
        })
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


def t212_to_yahoo(t212_ticker: str) -> str:
    """Convert NVDA_US_EQ / SPCXx.T to NVDA / SPCX for Yahoo price lookup."""
    return canonical_base_asset(t212_ticker)


def _chart(sym: str, rng: str = "1d", interval: str = "5m"):
    """Return (meta, bars) from Yahoo chart data; bars contain non-null closes.
    The intraday series is fresher than metadata. For a new listing, metadata can remain
    stale, e.g. SPCX volume=0 and an old price despite active trading."""
    if isinstance(sym, bool) or not isinstance(sym, str) or not sym.strip():
        return None, []
    sym_clean = sym.strip()
    qs = urllib.parse.urlencode({"range": str(rng or "1d").strip(), "interval": str(interval or "5m").strip()})
    status, body = http_get(YAHOO_CHART.format(sym=urllib.parse.quote(sym_clean, safe="")) + f"?{qs}",
                            headers=_UA)
    if status != 200 or not body:
        return None, []
    try:
        data = json.loads(body)
        if not isinstance(data, dict):
            return None, []
        chart = data.get("chart")
        if not isinstance(chart, dict):
            return None, []
        result = (chart.get("result") or [None])[0]
        if not isinstance(result, dict) or not result:
            return None, []
        meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
        ts = result.get("timestamp") or []
        indicators = result.get("indicators") if isinstance(result.get("indicators"), dict) else {}
        quote_list = indicators.get("quote") if isinstance(indicators.get("quote"), list) else []
        q = (quote_list[0] if quote_list and isinstance(quote_list[0], dict) else {}) or {}
        closes = q.get("close") or []
        vols = q.get("volume") or []
        bars = []
        for i, (t, c) in enumerate(zip(ts, closes)):
            if t is None or c is None or isinstance(t, bool) or isinstance(c, bool):
                continue
            try:
                t_f = float(t)
                c_f = float(c)
            except (TypeError, ValueError, OverflowError):
                continue
            if not (math.isfinite(t_f) and t_f > 0 and math.isfinite(c_f) and c_f > 0):
                continue
            raw_v = vols[i] if i < len(vols) else None
            v_clean = None
            if raw_v is not None and not isinstance(raw_v, bool):
                try:
                    v_f = float(raw_v)
                    if math.isfinite(v_f) and v_f >= 0:
                        v_clean = v_f
                except (TypeError, ValueError, OverflowError):
                    v_clean = None
            bars.append((int(t_f) if t_f.is_integer() else t_f, c_f, v_clean))
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
        raw_p = meta.get("regularMarketPrice")
        if raw_p is not None and not isinstance(raw_p, bool):
            try:
                p_f = float(raw_p)
                if math.isfinite(p_f) and p_f > 0:
                    return p_f
            except (TypeError, ValueError, OverflowError):
                pass
    return None


def trend_slope_pct(sym: str, bars: int = 12) -> float | None:
    """Return short-term OLS trend slope over the last `bars` five-minute closes.
    Normalize as price percentage PER BAR; negative means downtrend. Return None with
    insufficient data. Used as a DCA gate like Binance/Kraken to avoid falling markets."""
    if isinstance(bars, bool):
        bars_n = 12
    else:
        try:
            bars_n = max(4, int(bars))
        except (TypeError, ValueError, OverflowError):
            bars_n = 12
    _, bars_rows = _chart(sym)
    closes = [c for (_, c, _) in bars_rows if not isinstance(c, bool) and math.isfinite(float(c)) and float(c) > 0][-bars_n:]
    if len(closes) < max(4, bars_n // 2):
        return None
    n = len(closes)
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(closes) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0 or my == 0 or not math.isfinite(my):
        return None
    slope = sum((xs[i] - mx) * (closes[i] - my) for i in range(n)) / denom
    res = slope / my * 100.0     # percentage of price per five-minute bar
    return res if math.isfinite(res) else None


def get_usd_ron() -> float:
    """Return current USD/RON, falling back to 4.65 when the feed is unavailable."""
    rate = get_price_usd("USDRON=X")
    if rate is not None and not isinstance(rate, bool) and math.isfinite(rate) and rate > 1:
        return rate
    log("  ! curs USD/RON indisponibil, folosesc fallback 4.65")
    return 4.65


def get_eur_usd() -> float:
    """Return current EUR->USD (USD per EUR), falling back to 1.08."""
    rate = get_price_usd("EURUSD=X")
    if rate is not None and not isinstance(rate, bool) and math.isfinite(rate) and rate > 0.5:
        return rate
    log("  ! curs EUR/USD indisponibil, folosesc fallback 1.08")
    return 1.08


def check_market(sym: str) -> dict | None:
    """Return metadata with trading=True only when the symbol is actually trading.

    Avoid false positives from fixed-price, zero-volume IPO placeholders by requiring
    positive volume, a recent trade within 15 minutes, and an active market state.
    """
    meta, bars = _chart(sym)
    if meta is None and not bars:
        return None
    meta = meta or {}

    raw_price = meta.get("regularMarketPrice")
    price = None
    if raw_price is not None and not isinstance(raw_price, bool):
        try:
            p_f = float(raw_price)
            if math.isfinite(p_f) and p_f > 0:
                price = p_f
        except (TypeError, ValueError, OverflowError):
            price = None
    raw_vol = meta.get("regularMarketVolume")
    meta_vol = 0
    if raw_vol is not None and not isinstance(raw_vol, bool):
        try:
            v_f = float(raw_vol)
            if math.isfinite(v_f) and v_f >= 0:
                meta_vol = v_f
        except (TypeError, ValueError, OverflowError):
            meta_vol = 0
    raw_state = meta.get("marketState")
    state    = raw_state.strip().upper() if isinstance(raw_state, str) else ""
    last_ts  = meta.get("regularMarketTime")

    age_sec = None
    if last_ts is not None and not isinstance(last_ts, bool):
        try:
            ts_f = float(last_ts)
            if math.isfinite(ts_f) and ts_f > 0:
                age_sec = time.time() - ts_f
        except (TypeError, ValueError, OverflowError):
            pass

    # --- intraday SERIES signals robust to stale metadata ---
    series_age = series_vol = None
    series_price = None
    series_moved = False
    if bars:
        recent = bars[-6:]                       # approximately the last 30 minutes
        series_age = time.time() - float(bars[-1][0])
        series_price = bars[-1][1]
        series_vol = sum(v for _, _, v in recent if v) or 0
        cs = [c for _, c, _ in recent]
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
