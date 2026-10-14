#!/usr/bin/env python3
"""
market_data.py — Yahoo Finance market data: prices, FX rates, and real-trading detection.
"""

from __future__ import annotations

import json
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

from offline.backtests.datasets import drop_incomplete_last_bar, validate_dataset


def parse_interval_minutes(value: str) -> int:
    """Parse interval strings like '5m', '1h', '1d' into integer minutes."""
    units = {"m": 1, "h": 60, "d": 1440}
    try:
        minutes = int(value[:-1]) * units[value[-1].lower()]
    except (KeyError, ValueError, IndexError) as exc:
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
    query = {"range": range_, "interval": interval}
    day_range = re.fullmatch(r"([1-9][0-9]*)d", range_.lower())
    if day_range:
        end = int(time.time())
        query = {
            "period1": end - int(day_range.group(1)) * 24 * 60 * 60,
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
    result = (payload.get("chart", {}).get("result") or [None])[0]
    if not result:
        error = payload.get("chart", {}).get("error")
        raise RuntimeError(f"Yahoo returned no data for {symbol}: {error}")
    timestamps = result.get("timestamp") or []
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    records = []
    for index, timestamp in enumerate(timestamps):
        values = (
            quote.get("open", [])[index], quote.get("high", [])[index],
            quote.get("low", [])[index], quote.get("close", [])[index],
        )
        if None in values:
            continue
        records.append({
            "timestamp": int(timestamp),
            "open": float(values[0]),
            "high": float(values[1]),
            "low": float(values[2]),
            "close": float(values[3]),
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
    """Convert NVDA_US_EQ to NVDA for Yahoo price lookup."""
    return t212_ticker.split("_")[0]


def _chart(sym: str, rng: str = "1d", interval: str = "5m"):
    """Return (meta, bars) from Yahoo chart data; bars contain non-null closes.
    The intraday series is fresher than metadata. For a new listing, metadata can remain
    stale, e.g. SPCX volume=0 and an old price despite active trading."""
    status, body = http_get(YAHOO_CHART.format(sym=sym) + f"?range={rng}&interval={interval}",
                            headers=_UA)
    if status != 200 or not body:
        return None, []
    try:
        data = json.loads(body)
        result = (data.get("chart", {}).get("result") or [None])[0]
        if not result:
            return None, []
        meta = result.get("meta", {})
        ts = result.get("timestamp") or []
        q = ((result.get("indicators", {}).get("quote") or [{}])[0])
        closes = q.get("close") or []
        vols = q.get("volume") or []
        bars = [(t, c, (vols[i] if i < len(vols) else None))
                for i, (t, c) in enumerate(zip(ts, closes)) if c is not None]
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
        return meta.get("regularMarketPrice") or None
    return None


def trend_slope_pct(sym: str, bars: int = 12) -> float | None:
    """Return short-term OLS trend slope over the last `bars` five-minute closes.
    Normalize as price percentage PER BAR; negative means downtrend. Return None with
    insufficient data. Used as a DCA gate like Binance/Kraken to avoid falling markets."""
    _, bars_rows = _chart(sym)
    closes = [c for (_, c, _) in bars_rows][-bars:]
    if len(closes) < max(4, bars // 2):
        return None
    n = len(closes)
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(closes) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0 or my == 0:
        return None
    slope = sum((xs[i] - mx) * (closes[i] - my) for i in range(n)) / denom
    return slope / my * 100.0     # percentage of price per five-minute bar


def get_usd_ron() -> float:
    """Return current USD/RON, falling back to 4.65 when the feed is unavailable."""
    rate = get_price_usd("USDRON=X")
    if rate and rate > 1:
        return rate
    log("  ! curs USD/RON indisponibil, folosesc fallback 4.65")
    return 4.65


def get_eur_usd() -> float:
    """Return current EUR->USD (USD per EUR), falling back to 1.08."""
    rate = get_price_usd("EURUSD=X")
    if rate and rate > 0.5:
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

    price    = meta.get("regularMarketPrice")
    meta_vol = meta.get("regularMarketVolume") or 0
    state    = (meta.get("marketState") or "").upper()
    last_ts  = meta.get("regularMarketTime")

    age_sec = None
    if last_ts:
        try:
            age_sec = time.time() - float(last_ts)
        except (TypeError, ValueError):
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

    fresh_meta = age_sec is not None and age_sec < 15 * 60
    fresh_series = series_age is not None and series_age < 20 * 60
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
