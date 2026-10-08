# order_guard.py
"""Platform-agnostic profit guard.

Rule: do not BUY above the last SELL or SELL below the last BUY, subject to a minimum
margin. Decoupled from Binance, it accepts any `provider` implementing
`last_opposite_fill(symbol, order_type)` and an optional caller-computed `window_ref`
(for example, min(sell)/max(buy) from the Binance order cache). The SAME profit logic
therefore runs on every venue instead of being embedded only in bapi_placeorder.

Market-regime data sources and provider lookup are injected by callers, while the
shared market-regime types remain provider-neutral. Reference read failures from
provider.last_opposite_fill propagate so the caller can fail closed.
Returns True when placement is allowed and False when blocked.

The profit threshold is configured per venue in versioned, non-sensitive
`order_guard.conf`. A missing file or invalid value falls back to fail-safe 1.15%."""
import os
import time
import math
import json
from typing import Optional, Callable, Any, List, Tuple, Dict
import utils as u
from market_regime import MarketRegimeService, MarketRegimeDecision, MarketRegimeContext
from providers.strategy_executor import ProviderError

_DEFAULT_REGIME_SERVICE = MarketRegimeService()
_MARGINS = None   # cache: {provider_lower: percentage, "default": 1.15}
_MARGINS_FILE_MTIME: float = 0.0
_MARGINS_LAST_CHECK: float = 0.0
from geopolitical_order_guard import _GEO_SHADOW_NOTIFY_COOLDOWN as _SHADOW_NOTIFY_COOLDOWN


def _load_margins():
    """Read and cache `key = value` lines from order_guard.conf.
    The configuration file is the SINGLE source of truth. Checks file mtime
    periodically (every 5 seconds) to hot-reload config edits without restarting.
    The dictionary below is only a safety net for missing entries."""
    global _MARGINS, _MARGINS_FILE_MTIME, _MARGINS_LAST_CHECK
    now_ts = time.time()
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "order_guard.conf")
    if _MARGINS is not None and (now_ts - _MARGINS_LAST_CHECK) < 5.0:
        return _MARGINS
    _MARGINS_LAST_CHECK = now_ts
    try:
        mtime = os.path.getmtime(path) if os.path.exists(path) else 0.0
        if _MARGINS is not None and (_MARGINS_FILE_MTIME == 0.0 or mtime == _MARGINS_FILE_MTIME):
            _MARGINS_FILE_MTIME = mtime
            return _MARGINS
        _MARGINS_FILE_MTIME = mtime
    except Exception:
        if _MARGINS is not None:
            return _MARGINS
    m = {
        "default": 1.15,                       # fallback profit threshold (%)
        "default_window_h": 0.0,               # profit-guard window (hours); 0 uses only last_opposite_fill
        "default_max_daily_trades": 25,        # daily trade cap
        "default_safeback_sec": 14 * 24 * 3600 + 60,  # own-trade search window (seconds): 14 days
        "default_recent_transaction_sec": 180,   # anti-spam window (seconds)
        "default_buy_reference": 1.0,          # 1 = BUY must beat the historical sell reference
        "buy_window_mode": "dynamic",          # dynamic regime-aware lookback window
        "buy_window_bull_h": 8.0,              # Bull lookback: 4h - 12h
        "buy_window_flat_h": 24.0,             # Flat lookback: 24h - 48h
        "buy_window_bear_h": 72.0,             # Bear lookback: 48h - 168h
        "regime_context_max_age_sec": 120.0,   # Pre-computed regime context max age (seconds)
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "order_guard.conf")
    try:
        with open(path) as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if not line or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip().lower(); v = v.strip()
                try:                       # thresholds/hours/weights are floats; proxies are strings
                    m[k] = float(v)
                except ValueError:
                    m[k] = v
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[order_guard] invalid configuration ({e}) — using default 1.15")
    _MARGINS = m
    return m


def _provider_name(provider):
    if isinstance(provider, str):
        return provider
    return getattr(provider, "name", "") or ""


def buy_reference_mode(provider_name) -> str:
    """Return the buy reference mode: 'on', 'off', or 'dynamic'.

    Configuration key `<venue>_buy_reference`, falling back to `default_buy_reference`.
    '0', 0, False, 'off' -> 'off' (historical sell reference bypassed).
    '1', 1, True, 'on' -> 'on' (strict static lookback enforced).
    'dynamic' -> 'dynamic' (bypassed in confirmed BULL trend; enforced in BEAR/FLAT/UNKNOWN).
    """
    m = _load_margins()
    name = _provider_name(provider_name)
    key = name.lower() + "_buy_reference"
    val = m.get(key, m.get("default_buy_reference", 1.0))
    if isinstance(val, str):
        val_str = val.strip().lower()
        if val_str in ("dynamic", "adaptive"):
            return "dynamic"
        if val_str in ("0", "0.0", "false", "off", "no"):
            return "off"
        if val_str in ("1", "1.0", "true", "on", "yes"):
            return "on"
    try:
        return "off" if float(val) == 0.0 else "on"
    except (TypeError, ValueError):
        return "on"


def buy_reference_enabled(provider_name) -> bool:
    """Return whether a BUY must sit below the venue's historical sell reference.

    Compatibility wrapper around buy_reference_mode(provider_name) != 'off'."""
    return buy_reference_mode(provider_name) != "off"


def symbol_regime(
    symbol: str,
    provider=None,
    *,
    regime_service: Optional[MarketRegimeService] = None,
    now: Optional[float] = None,
    allow_fallback: bool = True,
    snapshot_resolver: Optional[Callable[[str, Optional[float]], Optional[dict]]] = None,
    provider_resolver: Optional[Callable[[Any], Any]] = None,
) -> MarketRegimeDecision:
    """Resolve the authoritative market regime decision for symbol via MarketRegimeService.

    Verifies freshness, observation timestamps, candle continuity, and provider fallback.
    Resolvers can be injected to avoid hidden imports and global state dependencies.
    """
    svc = regime_service or _DEFAULT_REGIME_SERVICE
    if not symbol:
        return svc.evaluator.unknown("missing_symbol")

    if hasattr(provider, "market_regime"):
        try:
            return provider.market_regime(symbol, allow_fallback=allow_fallback)
        except Exception:
            pass

    snap = None
    if snapshot_resolver is not None:
        try:
            snap = snapshot_resolver(symbol, now=now)
        except Exception:
            snap = None
    elif allow_fallback:
        # Graceful fallback for direct callers that did not inject a snapshot resolver
        try:
            import cacheManager as _cm
            mgr = _cm.get_short_trend_manager()
            snap = mgr.fresh_snapshot(symbol, now=now)
        except Exception:
            snap = None

    target_provider = provider
    if provider_resolver is not None:
        try:
            target_provider = provider_resolver(provider)
        except Exception:
            target_provider = None
    elif allow_fallback and isinstance(provider, str) and provider:
        try:
            from providers.market_api import api as _market_api
            target_provider = _market_api.provider_by_name(provider) or provider
        except Exception:
            target_provider = provider

    try:
        resolution = svc.resolve_with_evidence(
            target_provider,
            symbol,
            horizon="short",
            snapshot=snap,
            snapshot_max_age_seconds=svc.default_snapshot_max_age_seconds(),
            allow_fallback=allow_fallback,
            now=now,
        )
        return resolution.decision
    except Exception:
        return svc.evaluator.unknown("regime_resolution_failed")


def symbol_regime_context(
    symbol: str,
    provider=None,
    *,
    regime_service: Optional[MarketRegimeService] = None,
    now: Optional[float] = None,
    allow_fallback: bool = True,
    snapshot_resolver: Optional[Callable[[str, Optional[float]], Optional[dict]]] = None,
    provider_resolver: Optional[Callable[[Any], Any]] = None,
    trend_duration_seconds: float = 0.0,
    benchmark_symbol: Optional[str] = None,
) -> MarketRegimeContext:
    """Resolve and bundle MarketRegimeContext once for multi-guard placement pipelines."""
    now_ts = time.time() if now is None else float(now)
    decision = symbol_regime(
        symbol,
        provider=provider,
        regime_service=regime_service,
        now=now_ts,
        allow_fallback=allow_fallback,
        snapshot_resolver=snapshot_resolver,
        provider_resolver=provider_resolver,
    )
    provider_name = getattr(provider, "name", None) or (provider if isinstance(provider, str) else None)
    dur_sec = trend_duration_seconds
    if not dur_sec and symbol:
        dur_sec = _resolve_trend_duration(symbol)
    return MarketRegimeContext.from_decision(
        decision,
        evaluated_at=now_ts,
        symbol=symbol,
        provider=provider_name,
        trend_duration_seconds=dur_sec,
        benchmark_symbol=benchmark_symbol,
    )


def _symbol_trend(
    symbol: str,
    provider=None,
    regime_service=None,
    now=None,
    *,
    regime_context: Optional[MarketRegimeContext] = None,
    snapshot_resolver=None,
    provider_resolver=None,
) -> str:
    """Detect current macro/instant trend for symbol ('bull', 'bear', 'flat', 'unknown').

    Consumes the unified MarketRegimeDecision from MarketRegimeService.
    """
    if regime_context is not None:
        max_age = _load_margins().get("regime_context_max_age_sec", 120.0)
        if hasattr(regime_context, "is_valid_for"):
            if regime_context.is_valid_for(
                symbol=symbol, provider=provider, max_age_seconds=max_age, now=now,
                require_identity=True,
            ):
                return regime_context.resolved_trend
        else:
            return regime_context.resolved_trend
    try:
        decision = symbol_regime(
            symbol,
            provider=provider,
            regime_service=regime_service,
            now=now,
            snapshot_resolver=snapshot_resolver,
            provider_resolver=provider_resolver,
        )
        regime = decision.regime
        if regime == "sideways":
            return "flat"
        return regime
    except Exception:
        return "unknown"


def margin_for(provider_name):
    """Return the venue's configured minimum profit percentage, defaulting to 1.15."""
    m = _load_margins()
    return m.get((provider_name or "").lower(), m["default"])


def dynamic_buy_window_sec(
    symbol=None,
    provider=None,
    regime_service=None,
    now=None,
    *,
    resolved_trend=None,
    regime_context=None,
    snapshot_resolver=None,
    provider_resolver=None,
) -> float:
    """Calculate the dynamic lookback window (in seconds) for BUY reference:
    - BULL trend: 4h - 12h (default 8h)
    - BEAR trend: 48h - 168h (default 72h)
    - FLAT / UNKNOWN: 24h - 48h (default 24h)
    """
    m = _load_margins()
    try:
        bull_h = float(m.get("buy_window_bull_h", 8.0))
    except (ValueError, TypeError):
        bull_h = 8.0
    try:
        flat_h = float(m.get("buy_window_flat_h", 24.0))
    except (ValueError, TypeError):
        flat_h = 24.0
    try:
        bear_h = float(m.get("buy_window_bear_h", 72.0))
    except (ValueError, TypeError):
        bear_h = 72.0

    trend = resolved_trend
    if trend is None and regime_context is not None:
        max_age = _load_margins().get("regime_context_max_age_sec", 120.0)
        if not hasattr(regime_context, "is_valid_for") or regime_context.is_valid_for(
            symbol=symbol, provider=provider, max_age_seconds=max_age, now=now,
            require_identity=True,
        ):
            trend = getattr(regime_context, "resolved_trend", None)
    if trend not in {"bull", "bear", "flat", "sideways", "unknown"}:
        trend = (
            _symbol_trend(
                symbol,
                provider=provider,
                regime_service=regime_service,
                now=now,
                regime_context=regime_context,
                snapshot_resolver=snapshot_resolver,
                provider_resolver=provider_resolver,
            )
            if symbol
            else "unknown"
        )
    if trend == "bull":
        hours = max(4.0, min(12.0, bull_h))
    elif trend == "bear":
        hours = max(48.0, min(168.0, bear_h))
    else:  # flat, sideways, chop, or unknown
        hours = max(12.0, min(48.0, flat_h))
    return hours * 3600.0


def window_for(
    provider_name,
    symbol=None,
    order_type=None,
    *,
    regime_context=None,
) -> float:
    """Return the reference window in seconds for the given venue and order side.

    For BUY orders when dynamic windowing is active (via venue buy_reference='dynamic'
    or buy_window_mode='dynamic'), returns dynamic_buy_window_sec(symbol, provider=name)
    scaling lookback from 4h-12h in bull to 48h-168h in bear.
    For SELL orders or static venues, returns `<venue>_window_h` (or default_window_h) in seconds.
    """
    name = _provider_name(provider_name)
    m = _load_margins()
    side = (order_type or "").upper()

    if side == "BUY":
        mode = buy_reference_mode(name)
        if mode == "off":
            return 0.0
        win_mode = str(m.get("buy_window_mode", "dynamic")).strip().lower()
        if mode == "dynamic" or win_mode in ("dynamic", "adaptive"):
            return dynamic_buy_window_sec(
                symbol,
                provider=name,
                regime_context=regime_context,
            )

    key = (name.lower() if name else "") + "_window_h"
    hours = m.get(key, m.get("default_window_h", 0.0))
    try:
        hours_val = float(hours)
    except (ValueError, TypeError):
        hours_val = 0.0
    return hours_val * 3600.0


def weight_proxy_for(provider_name):
    """Return the trend/Gaussian-weight proxy symbol used when the current symbol lacks
    its own long trend (e.g. HYPE -> BTC until data exists). Uses `<venue>_weight_proxy`
    or `default_weight_proxy`; None falls back to weight 0.03."""
    m = _load_margins()
    return m.get((provider_name or "").lower() + "_weight_proxy", m.get("default_weight_proxy"))


def window_reference(provider, symbol, order_type, window_s):
    """Return a time-windowed reference from opposite-side orders/fills: minimum SELL
    for a BUY or maximum BUY for a SELL. Return None for an empty or disabled window.
    Read errors propagate so the caller fails closed. Non-positive prices are ignored.
    This is the platform-agnostic equivalent of Binance tier 1."""
    if not window_s or window_s <= 0:
        return None
    opp = "SELL" if order_type.upper() == "BUY" else "BUY"
    recent = provider.get_orders(symbol, opp, window_s)
    if recent is None:
        raise ProviderError(f"window_reference({symbol}): failed to read order history")
    prices = [float(o.get("price") or 0) for o in recent if float(o.get("price") or 0) > 0]
    if not prices:
        return None
    return min(prices) if order_type.upper() == "BUY" else max(prices)


def chop_weight_for(provider_name):
    """Return the venue's conservative chop fallback weight (default 0.03)."""
    m = _load_margins()
    key = (provider_name or "").lower() + "_chop_weight"
    val = m.get(key, m.get("default_chop_weight", 0.03))
    try:
        f = float(val)
        return f if math.isfinite(f) and f > 0 else 0.03
    except (TypeError, ValueError):
        return 0.03


def resolve_trade_weight(symbol, order_type, provider_name=None, *, default_weight=None, raise_on_error=False):
    """Resolve trade weight using priceAnalysis Gaussian curve, proxy, or conservative fallback.

    1. Gaussian weight for symbol when long trend exists.
    2. Proxy Gaussian weight (e.g. BTCUSDC) when symbol has no trend.
    3. Conservative chop fallback (default 0.03 or venue-configured chop weight).
    """
    import math

    def _gauss(sym):
        try:
            import priceAnalysis as pa
            val = pa.get_weight_for_cash_permission_at_quant_time(sym, order_type)
            if val is not None:
                fval = float(val)
                if not math.isfinite(fval) or not 0 < fval <= 1:
                    if raise_on_error and sym == symbol:
                        from providers.strategy_executor import SubmissionRefused
                        raise SubmissionRefused("invalid_weight_policy_weight")
                    return None
                return fval
            return None
        except Exception as e:
            if raise_on_error and sym == symbol:
                raise
            print(f"[WEIGHT] {sym}: cannot compute the gauss value ({e})")
            return None

    weight = _gauss(symbol)
    if weight is None:
        proxy = weight_proxy_for(provider_name)
        if proxy and proxy != symbol:
            try:
                pw = _gauss(proxy)
                if pw is not None:
                    weight = pw
                    print(f"[WEIGHT] {symbol}: no trend of its own -> proxy {proxy} (weight={weight})")
            except Exception as e:
                print(f"[WEIGHT] {symbol}: proxy {proxy} lookup failed ({e})")

    if weight is None:
        if default_weight is None:
            default_weight = chop_weight_for(provider_name)
        weight = default_weight
        print(f"[WEIGHT] {symbol}: no trend or proxy -> fallback weight={weight}")

    return float(weight)


def compute_weight_capped_qty(weight, price, required_qty, available_qty, traded_24h_value):
    """Cap per-order quantity using Gaussian/chop weight and 24h traded quote reference."""
    price = float(price)
    available = float(available_qty)
    required = float(required_qty)
    traded = float(traded_24h_value)

    total_ref = traded + available * price
    max_trade_value = total_ref * float(weight)
    remaining_value = max(0.0, max_trade_value - traded)
    remaining_qty = remaining_value / price if price > 0 else 0.0
    return min(required, remaining_qty)


def weight_limit(provider, symbol, order_type, price, required_qty, *, available_qty):
    """Cap per-order quantity using the Gaussian curve, the platform-agnostic equivalent
    of bapi.apply_weight_limit. Allocate tradable value by trend position so the whole
    amount is not bought or sold at once. priceAnalysis supplies the Gaussian weight;
    provider.get_orders supplies 24-hour traded value. The shared quantity decision must
    supply the side-aware balance, which this guard does not re-read. Return the smaller
    of requested and permitted quantity. Errors propagate so the caller fails closed."""
    provider_name = getattr(provider, "name", "")
    weight = resolve_trade_weight(symbol, order_type, provider_name)
    recent = provider.get_orders(symbol, order_type, 86400)
    if recent is None:
        raise ProviderError(f"weight_limit({symbol}): order history unavailable")
    traded_value = sum(float(o.get("price", 0)) * float(o.get("qty", o.get("quantity", 0))) for o in recent)
    available = float(available_qty)
    adjusted = compute_weight_capped_qty(weight, price, required_qty, available, traded_value)
    total_ref = traded_value + available * price
    max_trade_value = total_ref * weight
    remaining_value = max(0.0, max_trade_value - traded_value)
    print(f"[WEIGHT] {order_type} {symbol}: weight={weight} traded24h={traded_value:.2f} "
          f"avail={available:.6f} max={max_trade_value:.2f} remaining={remaining_value:.2f} "
          f"cerut={required_qty:.6f} -> {adjusted:.6f}")
    return adjusted


def max_daily_trades_for(provider_name):
    """Return the venue's daily trade cap from order_guard.conf or the seed fallback."""
    m = _load_margins()
    key = (provider_name or "").lower() + "_max_daily_trades"
    return int(m.get(key, m["default_max_daily_trades"]))


def safeback_sec_for(provider_name):
    """Return the per-venue trade-search window in seconds for the daily cap."""
    m = _load_margins()
    key = (provider_name or "").lower() + "_safeback_sec"
    return float(m.get(key, m["default_safeback_sec"]))


def recent_transaction_sec_for(provider_name):
    """Return the anti-spam window that rejects a recent same-symbol, same-side order."""
    m = _load_margins()
    key = (provider_name or "").lower() + "_recent_transaction_sec"
    return float(m.get(key, m["default_recent_transaction_sec"]))


def daily_limit_guard(provider, symbol, order_type, max_daily_trades=None,
                      safeback_sec=None, recent_transaction_sec=None):
    """Enforce the daily cap and anti-spam policy as the SINGLE implementation.
    Since July 30, Binance bapi_placeorder.if_place_safe_order delegates here instead of
    maintaining a second inline version; Kraken and Hyperliquid use it through
    Instrument.place(). provider.get_orders supplies same-side orders from safeback_sec.
    Exceeding max_daily_trades per day returns "daily_limit"; a record inside
    recent_transaction_sec returns "recent_transaction".

    backdays = math.ceil(safeback_sec/86400), rounded UP to preserve the historical
    Binance formula and its effective threshold rather than using simple division.

    Return (True, None) or (False, reason). Read errors propagate so the caller can fail
    closed, consistently with profit_guard and weight_limit."""
    name = _provider_name(provider)
    max_daily_trades = max_daily_trades if max_daily_trades is not None else max_daily_trades_for(name)
    safeback_sec = float(safeback_sec if safeback_sec is not None else safeback_sec_for(name))
    recent_transaction_sec = float(recent_transaction_sec if recent_transaction_sec is not None
                                   else recent_transaction_sec_for(name))
    order_type = order_type.upper()
    try:
        trades = provider.get_orders(symbol, order_type, safeback_sec)
    except ProviderError as exc:
        print(f"[DAILY-LIMIT] {order_type} {symbol}: history unavailable ({exc}) -> BLOCKED")
        return False, "history_unavailable"
    except Exception as exc:
        print(f"[DAILY-LIMIT] {order_type} {symbol}: history read failed ({exc}) -> BLOCKED")
        return False, "history_unavailable"
    if trades is None:
        print(f"[DAILY-LIMIT] {order_type} {symbol}: history unavailable -> BLOCKED")
        return False, "history_unavailable"
    backdays = max(math.ceil(safeback_sec / 86400.0), 1)
    if len(trades) / backdays > max_daily_trades:
        print(f"[DAILY-LIMIT] {order_type} {symbol}: {len(trades)} trades in "
              f"{safeback_sec/3600:.1f}h, a cap of {max_daily_trades}/day -> BLOCKED")
        return False, "daily_limit"
    cutoff_ms = time.time() * 1000 - recent_transaction_sec * 1000
    for t in trades:
        ts = t.get("timestamp")
        if ts is not None and float(ts) >= cutoff_ms:
            print(f"[DAILY-LIMIT] {order_type} {symbol}: recent trade "
                  f"(<{recent_transaction_sec:.0f}s) -> BLOCKED")
            return False, "recent_transaction"
    return True, None


_whale_collector = None
_orderbook_collector = None
_derivatives_collector = None


def get_whale_collector():
    global _whale_collector
    if _whale_collector is not None:
        return _whale_collector
    from external_order_guard import get_whale_collector as _gwc
    return _gwc()


def get_orderbook_collector():
    global _orderbook_collector
    if _orderbook_collector is not None:
        return _orderbook_collector
    from external_order_guard import get_orderbook_collector as _goc
    return _goc()


def get_derivatives_collector():
    global _derivatives_collector
    if _derivatives_collector is not None:
        return _derivatives_collector
    from external_order_guard import get_derivatives_collector as _gdc
    return _gdc()


def _normalize_base_asset(sym: str) -> str:
    s = (sym or "").strip().upper()
    for q in ("USDC", "USDT", "FDUSD", "USD", "EUR", "RON", "GBP"):
        if s.endswith(q) and len(s) > len(q):
            return s[:-len(q)]
    return s


def _read_cached_trend_duration(symbol: str) -> float:
    try:
        cachedb_dir = os.environ.get("MPTRADE_CACHEDB_DIR", "cachedb")
        p = os.path.join(cachedb_dir, "cache_price_long_trend.json")
        if not os.path.exists(p):
            return 0.0
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_items = data.get("items", {})
        items = all_items.get(symbol.upper())
        if not items:
            su_base = _normalize_base_asset(symbol)
            for k, v in all_items.items():
                if _normalize_base_asset(k) == su_base:
                    items = v
                    break
        if items and isinstance(items, list):
            for entry in reversed(items):
                if isinstance(entry, dict) and entry.get("duration_seconds") is not None:
                    dur = float(entry.get("duration_seconds", 0.0) or 0.0)
                    if dur > 0:
                        return dur
    except Exception:
        pass
    return 0.0


def _resolve_trend_duration(symbol: str) -> float:
    """Resolve trend duration in seconds from memory/cache files once."""
    if not symbol:
        return 0.0
    try:
        import cacheManager as cm
        trend_meta = getattr(cm, "read_long_term_trend_file", lambda s: None)(symbol)
        if trend_meta and "duration_seconds" in trend_meta:
            dur = float(trend_meta["duration_seconds"] or 0.0)
            if dur > 0:
                return dur
    except Exception:
        pass
    return _read_cached_trend_duration(symbol)


def _read_cached_price_history(symbol: str, window_seconds: float = 7200.0) -> Optional[List[Tuple[float, float]]]:
    """Read rolling price history from local cache for parabolic surge evaluation.

    Prefers the authoritative 24h per-symbol cache (cachedb/cache_24price_{symbol}.json)
    which captures every high-resolution tick, and falls back to cache_prices_multi.json
    with exact normalized base-symbol matching for cross-venue assets (e.g. HYPE on Kraken).
    Never uses substring matching so e.g. ETHFI does not accidentally match ETH.
    """
    su = (symbol or "").strip().upper()
    cachedb_dir = os.environ.get("MPTRADE_CACHEDB_DIR", "cachedb")
    now_ts = time.time()
    cutoff_ts = now_ts - window_seconds
    history: List[Tuple[float, float]] = []

    # 1. Primary: dedicated 24h cache (cache_24price_{symbol}.json)
    p24 = os.path.join(cachedb_dir, f"cache_24price_{su}.json")
    if not os.path.exists(p24):
        su_base = _normalize_base_asset(su)
        p24_base = os.path.join(cachedb_dir, f"cache_24price_{su_base}.json")
        if os.path.exists(p24_base):
            p24 = p24_base
    if os.path.exists(p24):
        try:
            with open(p24, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw_items = data.get("items", {}).get(su)
            if not raw_items:
                su_base = _normalize_base_asset(su)
                raw_items = data.get("items", {}).get(su_base, [])
            for item in raw_items:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    try:
                        ts_raw = float(item[0])
                        ts_sec = ts_raw / 1000.0 if ts_raw > 1e11 else ts_raw
                        px = float(item[1])
                        if ts_sec >= cutoff_ts and px > 0:
                            history.append((ts_sec, px))
                    except (ValueError, TypeError):
                        continue
        except Exception:
            pass

    # 2. Fallback: shared multi-symbol cache (cache_prices_multi.json)
    if not history:
        p_multi = os.path.join(cachedb_dir, "cache_prices_multi.json")
        if os.path.exists(p_multi):
            try:
                with open(p_multi, "r", encoding="utf-8") as f:
                    data = json.load(f)
                all_items = data.get("items", {})
                raw_items = all_items.get(su)
                if not raw_items:
                    su_base = _normalize_base_asset(su)
                    # Support exact normalized base asset keys (e.g. HYPE for HYPEUSD or HYPEUSDC)
                    raw_items = all_items.get(su_base)
                    if not raw_items:
                        for k, v in all_items.items():
                            if k and _normalize_base_asset(k) == su_base:
                                raw_items = v
                                break
                if raw_items:
                    for item in raw_items:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            try:
                                ts_raw = float(item[0])
                                ts_sec = ts_raw / 1000.0 if ts_raw > 1e11 else ts_raw
                                px = float(item[1])
                                if ts_sec >= cutoff_ts and px > 0:
                                    history.append((ts_sec, px))
                            except (ValueError, TypeError):
                                continue
            except Exception:
                pass

    return history if history else None


def check_intelligence_guards(
    provider,
    symbol: str,
    order_type: str,
    price: float,
    *,
    price_history=None,
    trend_duration_seconds: float = 0.0,
    regime_context=None,
    qty: Optional[float] = None,
    notional_eur: Optional[float] = None,
    is_stop_loss: bool = False,
    now: Optional[float] = None,
) -> tuple[bool, str, float]:
    """Evaluate intelligence guards across all active pillars.

    Supports staged rollouts via order_guard.conf:
    - 'intelligence_guards_mode': 'off' | 'shadow' | 'enforce' (Pillar 1)
    - 'gemini_guard_mode': 'off' | 'shadow' | 'enforce' (Pillar 3, purchases >= 1000 EUR)
    - 'geopolitical_guard_mode': 'off' | 'shadow' | 'enforce' (Pillar 4, Black Swan Shield)
    - 'whale_guard_mode': 'off' | 'shadow' | 'enforce' (Pillar 2, Whale Flow & OI Divergence)
    - 'orderbook_wall_guard_mode': 'off' | 'shadow' | 'enforce' (Pillar 2, Orderbook Depth & Ask Walls)
    - 'funding_guard_mode': 'off' | 'shadow' | 'enforce' (Pillar 2, Derivatives Funding Crowding)

    Returns:
        (allowed: bool, reason: str, suggested_scale: float)
    """
    if is_stop_loss:
        return True, "stop_loss_exempt", 1.0

    valid_context = False
    if regime_context is not None:
        max_age = _load_margins().get("regime_context_max_age_sec", 120.0)
        if hasattr(regime_context, "is_valid_for"):
            valid_context = regime_context.is_valid_for(
                symbol=symbol, provider=provider, max_age_seconds=max_age, now=now,
                require_identity=True,
            )
        else:
            valid_context = True

    m = _load_margins()
    mode = str(m.get("internal_guard_mode", m.get("intelligence_guards_mode", "shadow"))).strip().lower()
    if mode in ("off", "0", "disabled"):
        return True, "intelligence_guards_off", 1.0

    side = (order_type or "").upper()
    if side not in ("BUY", "SELL"):
        return True, "not_supported_side", 1.0

    computed_notional = notional_eur
    if computed_notional is None and qty is not None and price > 0:
        computed_notional = price * qty

    if side == "SELL":
        sell_min_notional = float(m.get("sell_guard_min_notional_eur", 1000.0))
        if computed_notional is None or computed_notional < sell_min_notional:
            return True, "small_sell_exempt", 1.0

    # Check if expensive static analysis is already cached on regime_context
    cached_res = getattr(regime_context, "_intelligence_decision", None) if valid_context else None
    if cached_res is None:
        cached_res = _evaluate_intelligence_guards_raw(
            provider,
            symbol,
            order_type,
            price,
            price_history=price_history,
            trend_duration_seconds=trend_duration_seconds,
            regime_context=regime_context if valid_context else None,
            qty=qty,
            notional_eur=computed_notional,
            is_stop_loss=is_stop_loss,
            now=now,
        )
        if valid_context:
            try:
                object.__setattr__(regime_context, "_intelligence_decision", cached_res)
                object.__setattr__(regime_context, "_intelligence_price", price)
                object.__setattr__(regime_context, "_intelligence_notional", computed_notional)
            except Exception:
                pass

    static_allowed, static_reason, static_scale = cached_res
    if not static_allowed:
        # Rejected by baseline or macro guard (e.g. Black Swan, Whale divergence, Ask Wall)
        return False, static_reason, 0.0

    if not valid_context:
        return cached_res

    effective_scale = static_scale
    active_reason = static_reason

    # Dynamic Re-evaluation for cached context:
    # 1. Dynamic Parabolic Surge Guard (Anti-FOMO spike check evaluated on BUY only)
    if side == "BUY" and price > 0:
        from intelligence.internal.guards.parabolic_guard import ParabolicSurgeGuard
        from intelligence.internal.guards.guard_decision import BrakeAction
        surge_pct = float(m.get("parabolic_surge_pct", 15.0))
        pullback_pct = float(m.get("parabolic_pullback_pct", 2.0))
        history = price_history
        if history is None:
            history = _read_cached_price_history(symbol, window_seconds=7200.0)
        if history:
            p_guard = ParabolicSurgeGuard(surge_threshold_pct=surge_pct, pullback_required_pct=pullback_pct)
            p_dec = p_guard.check(symbol, side, price, price_history=history, now=now)
            if not p_dec.allowed:
                prefix = "[INTELLIGENCE_GUARD_SHADOW]" if mode == "shadow" else "[INTELLIGENCE_GUARD_ENFORCE]"
                print(f"{prefix} {side} {symbol} @ {price}: {p_dec.reason} (brake={p_dec.brake_action})")
                if mode == "enforce":
                    return False, p_dec.reason, 0.0

    # Resolve trend duration early so both LLM guard and Weibull guard have access
    dur_sec = trend_duration_seconds or (getattr(regime_context, "trend_duration_seconds", 0.0) if valid_context else 0.0)
    if not dur_sec and symbol:
        dur_sec = _resolve_trend_duration(symbol)

    # 2. Dynamic LLM Pre-Trade Guard if notional crossed the threshold
    llm_mode = str(
        m.get("llm_guard_mode",
        m.get("high_stake_guard_mode",
        m.get("macro_stake_guard_mode",
        m.get("macro_guard_mode",
        m.get("gemini_guard_mode", "shadow")))))
    ).strip().lower()
    if llm_mode not in ("off", "0", "disabled"):
        min_notional = float(
            m.get("llm_min_notional_eur",
            m.get("high_stake_min_notional_eur",
            m.get("macro_stake_min_notional_eur",
            m.get("gemini_min_notional_eur", 1000.0))))
        )
        if computed_notional is not None and computed_notional >= min_notional:
            cached_gemini_notional = getattr(regime_context, "_gemini_evaluated_notional", None)
            if cached_gemini_notional is None:
                cached_gemini_notional = getattr(regime_context, "_intelligence_notional", None)
            notional_jump = (
                cached_gemini_notional is not None
                and (computed_notional - cached_gemini_notional) / max(cached_gemini_notional, 1.0) > 0.20
            )
            # If notional wasn't evaluated for high-stake in cached_res, or has jumped significantly (>20%) from evaluated baseline, re-evaluate now
            if cached_gemini_notional is None or cached_gemini_notional < min_notional or notional_jump:
                from intelligence.internal.guards.guard_decision import BrakeAction
                from intelligence.sentiment.guards.high_stake_guard import HighStakeGuard
                timeout_sec = float(m.get("llm_timeout_sec", m.get("high_stake_timeout_sec", m.get("gemini_timeout_sec", 25.0))))
                fallback = str(m.get("llm_fallback", m.get("high_stake_fallback", m.get("gemini_fallback", "allow")))).strip().lower()
                llm_guard = HighStakeGuard(min_notional_eur=min_notional, timeout_sec=timeout_sec, fallback_action=fallback)
                try:
                    g_dec = llm_guard.check(
                        symbol,
                        side,
                        price,
                        qty if qty is not None else 1.0,
                        notional_eur=computed_notional,
                        regime_context=regime_context if valid_context else None,
                        dur_sec=dur_sec,
                        is_stop_loss=is_stop_loss,
                    )
                except TypeError:
                    g_dec = llm_guard.check(
                        symbol,
                        side,
                        price,
                        qty if qty is not None else 1.0,
                        notional_eur=computed_notional,
                    )
                if valid_context:
                    try:
                        object.__setattr__(regime_context, "_gemini_evaluated_notional", computed_notional)
                        object.__setattr__(regime_context, "_gemini_decision", (g_dec.allowed, g_dec.reason, g_dec.suggested_scale))
                    except Exception:
                        pass
                if not g_dec.allowed or g_dec.brake_action == BrakeAction.DOWNSCALE_QTY:
                    prefix = "[LLM_GUARD_SHADOW]" if llm_mode == "shadow" else "[LLM_GUARD_ENFORCE]"
                    print(f"{prefix} {side} {symbol} €{computed_notional:.2f}: {g_dec.reason} (brake={g_dec.brake_action}, suggested_scale={g_dec.suggested_scale})")
                    if llm_mode == "enforce":
                        if not g_dec.allowed:
                            return False, g_dec.reason, 0.0
                        effective_scale = min(effective_scale, g_dec.suggested_scale)
                        active_reason = g_dec.reason
            else:
                # Re-apply cached Gemini decision for unchanged notional baseline
                cached_g_dec = getattr(regime_context, "_gemini_decision", None) if valid_context else None
                if cached_g_dec is not None:
                    g_allowed, g_reason, g_scale = cached_g_dec
                    if not g_allowed or g_scale < 1.0:
                        if llm_mode == "enforce":
                            if not g_allowed:
                                return False, g_reason, 0.0
                            effective_scale = min(effective_scale, g_scale)
                            active_reason = g_reason

    # 3. Dynamic Weibull trend exhaustion check if duration provided or resolved (BUY entries only)
    if side == "BUY" and dur_sec and dur_sec > 0:
        from intelligence.internal.guards.exhaustion_guard import WeibullExhaustionGuard
        from intelligence.internal.guards.guard_decision import BrakeAction
        e_policy = str(m.get("weibull_exhaustion_policy", "downscale")).strip().lower()
        e_scale = float(m.get("weibull_exhausted_scale", 0.25))
        e_p90 = float(m.get("weibull_p90_days", 7.0))
        e_guard = WeibullExhaustionGuard(default_p90_days=e_p90, policy=e_policy, exhausted_scale=e_scale)
        e_dec = e_guard.check(symbol, side, trend_duration_seconds=dur_sec)
        if e_dec.brake_action != BrakeAction.NONE:
            prefix = "[INTELLIGENCE_GUARD_SHADOW]" if mode == "shadow" else "[INTELLIGENCE_GUARD_ENFORCE]"
            print(f"{prefix} {side} {symbol} @ {price}: {e_dec.reason} (brake={e_dec.brake_action}, suggested_scale={e_dec.suggested_scale})")
            if mode == "enforce":
                if not e_dec.allowed:
                    return False, e_dec.reason, 0.0
                effective_scale = min(effective_scale, e_dec.suggested_scale)
                active_reason = e_dec.reason

    final_res = (True, active_reason if effective_scale < 1.0 else "ok", effective_scale)
    if valid_context:
        try:
            object.__setattr__(regime_context, "_intelligence_price", price)
            if computed_notional is not None:
                object.__setattr__(regime_context, "_intelligence_notional", computed_notional)
            object.__setattr__(regime_context, "suggested_scale", effective_scale)
            object.__setattr__(regime_context, "intelligence_reason", active_reason)
        except Exception:
            pass
    return final_res


def _evaluate_intelligence_guards_raw(
    provider,
    symbol: str,
    order_type: str,
    price: float,
    *,
    price_history=None,
    trend_duration_seconds: float = 0.0,
    regime_context=None,
    qty: Optional[float] = None,
    notional_eur: Optional[float] = None,
    is_stop_loss: bool = False,
    now: Optional[float] = None,
) -> tuple[bool, str, float]:
    m = _load_margins()
    mode = str(m.get("internal_guard_mode", m.get("intelligence_guards_mode", "shadow"))).strip().lower()
    if mode in ("off", "0", "disabled"):
        return True, "intelligence_guards_off", 1.0

    if is_stop_loss:
        return True, "stop_loss_exempt", 1.0

    side = (order_type or "").upper()
    if side not in ("BUY", "SELL"):
        return True, "not_supported_side", 1.0

    computed_notional = notional_eur
    if computed_notional is None and qty is not None and price > 0:
        computed_notional = price * qty

    if side == "SELL":
        sell_min_notional = float(m.get("sell_guard_min_notional_eur", 1000.0))
        if computed_notional is None or computed_notional < sell_min_notional:
            return True, "small_sell_exempt", 1.0

    effective_scale = 1.0
    active_reason = "ok"

    # 1. Internal Quantitative Mathematical Guards (Pillar 1: Parabolic Surge, Weibull Exhaustion)
    if side == "BUY":
        from internal_order_guard import check_internal_order_guards
        int_ok, int_reason, int_scale = check_internal_order_guards(
            symbol=symbol,
            side=side,
            price=price,
            price_history=price_history,
            trend_duration_seconds=trend_duration_seconds,
            regime_context=regime_context,
            margins=m,
            now=now,
        )
        if not int_ok:
            return False, int_reason, 0.0
        if int_scale < 1.0:
            effective_scale = min(effective_scale, int_scale)
            active_reason = int_reason

    # 2. External Microstructure & Derivatives Guards (Pillar 2: Orderbook Depth, Funding, Whale Flow)
    from external_order_guard import check_external_order_guards
    ext_ok, ext_reason, ext_scale = check_external_order_guards(
        symbol=symbol,
        side=side,
        price=price,
        qty=qty,
        notional_eur=computed_notional,
        is_stop_loss=is_stop_loss,
        margins=m,
        now=now,
    )
    if not ext_ok:
        return False, ext_reason, 0.0
    if ext_scale < 1.0:
        effective_scale = min(effective_scale, ext_scale)
        active_reason = ext_reason

    # 3. Intelligence & LLM Guards (Pillar 3: High-Stake Pre-Trade LLM + Geopolitical Black Swan Shield)
    from intelligence_order_guard import check_intelligence_order_guards
    intel_ok, intel_reason, intel_scale = check_intelligence_order_guards(
        provider=provider,
        symbol=symbol,
        order_type=order_type,
        price=price,
        qty=qty,
        notional_eur=computed_notional,
        is_stop_loss=is_stop_loss,
        regime_context=regime_context,
        margins=m,
        now=now,
    )
    if not intel_ok:
        return False, intel_reason, 0.0
    if intel_scale < 1.0:
        effective_scale = min(effective_scale, intel_scale)
        active_reason = intel_reason

    return True, active_reason if effective_scale < 1.0 else "ok", effective_scale


def check_math_profit_reference(
    provider,
    symbol: str,
    order_type: str,
    price: float,
    profit_percentage: float,
    window_ref=None,
    *,
    regime_context=None,
) -> bool:
    """Evaluate deterministic price reference and margin percentage thresholds.

    Pure mathematical calculation (0 tokens, deterministic, no AI or macro push alerts).
    """
    order_type = order_type.upper()
    provider_name = _provider_name(provider)
    if order_type == "BUY":
        mode = buy_reference_mode(provider_name)
        if mode == "off":
            print(f"[GUARD] BUY {symbol}: the historical sell reference is off for this venue "
                  f"(order_guard.conf); price {price} is not compared with past sells")
            return True
        elif mode == "dynamic":
            trend = None
            if regime_context is not None:
                max_age = _load_margins().get("regime_context_max_age_sec", 120.0)
                if not hasattr(regime_context, "is_valid_for") or regime_context.is_valid_for(
                    symbol=symbol, provider=provider_name, max_age_seconds=max_age,
                    require_identity=True,
                ):
                    trend = getattr(regime_context, "resolved_trend", None)
            if trend is None:
                trend = _symbol_trend(symbol, provider=provider)
            dyn_window_s = dynamic_buy_window_sec(
                symbol,
                provider=provider,
                resolved_trend=trend,
                regime_context=regime_context,
            )
            dyn_hours = dyn_window_s / 3600.0
            def _valid_pos_float(val):
                if val is None:
                    return None
                try:
                    f = float(val)
                    return f if math.isfinite(f) and f > 0 else None
                except (TypeError, ValueError):
                    return None

            pos_window_ref = _valid_pos_float(window_ref)
            if pos_window_ref is not None:
                diff = u.value_diff_to_percent(pos_window_ref, price)
                if diff < profit_percentage:
                    # In dynamic mode, verify whether this reference actually sits within the dynamic window (dyn_window_s)
                    if hasattr(provider, "get_orders"):
                        try:
                            recent = provider.get_orders(symbol, "SELL", dyn_window_s)
                        except (ProviderError, Exception) as exc:
                            print(f"[GUARD] BUY {symbol}: dynamic window history unavailable ({exc}) -> BLOCKED")
                            return False
                        if recent is None:
                            print(f"[GUARD] BUY {symbol}: dynamic window history unavailable -> BLOCKED")
                            return False
                        recent_prices = []
                        for o in recent:
                            try:
                                px = float(o.get("price", 0.0) or 0.0)
                                if px > 0:
                                    recent_prices.append(px)
                            except (ValueError, TypeError):
                                continue
                        if not recent_prices:
                            print(f"[GUARD] BUY {symbol}: dynamic window ({dyn_hours:.1f}h, trend='{trend}') has no fills; "
                                  f"anchor {pos_window_ref} from older period bypassed")
                            return True
                    elif trend == "bull" and not hasattr(provider, "get_orders"):
                        print(f"[GUARD] BUY {symbol}: dynamic mode bypassed historical sell reference "
                              f"during confirmed BULL trend; price {price} permitted")
                        return True
                    print(f"[GUARD] BUY {symbol}: dynamic mode ({dyn_hours:.1f}h window, trend='{trend}') "
                          f"found recent sell ref {pos_window_ref}, price {price}, diff {diff:.2f}%, threshold {profit_percentage}%")
                    print(f"Percentage difference ({diff:.2f}%) below threshold {profit_percentage}%. "
                          f"The BUY order is BLOCKED.")
                    return False
                print(f"[GUARD] BUY {symbol}: dynamic mode ({dyn_hours:.1f}h window, trend='{trend}') "
                      f"recent sell ref {pos_window_ref}, price {price}, diff {diff:.2f}% >= {profit_percentage}%. Permitted.")
                return True
            else:
                print(f"[GUARD] BUY {symbol}: dynamic mode ({dyn_hours:.1f}h window, trend='{trend}') "
                      f"has no recent sell reference; price {price} permitted")
                return True
    has_window = window_for(provider_name, symbol, order_type, regime_context=regime_context) > 0
    raw_ref = window_ref if window_ref is not None else (
        None if has_window else (
            provider.last_opposite_fill(symbol, order_type) if hasattr(provider, "last_opposite_fill") else None
        )
    )
    def _parse_ref(val):
        if val is None:
            return None
        try:
            f = float(val)
            return f if math.isfinite(f) and f > 0 else None
        except (TypeError, ValueError):
            return None

    ref = _parse_ref(raw_ref)
    if ref is None:
        return True
    if order_type == "BUY":
        diff = u.value_diff_to_percent(ref, price)   # (ref_SELL - BUY_price) / ref_SELL
    else:
        diff = u.value_diff_to_percent(price, ref)   # (SELL_price - ref_BUY) / SELL_price
    src = "the window" if window_ref is not None else "the provider"
    print(f"[GUARD] {order_type} {symbol}: ref {ref} ({src}), price {price}, "
          f"diff {diff:.2f}%, threshold {profit_percentage}%")
    if diff < profit_percentage:
        print(f"Percentage difference ({diff:.2f}%) below threshold {profit_percentage}%. "
              f"The {order_type} order is BLOCKED.")
        return False
    return True


_check_math_profit_reference = check_math_profit_reference


def profit_guard(
    provider,
    symbol,
    order_type,
    price,
    profit_percentage,
    window_ref=None,
    *,
    regime_context=None,
    qty: Optional[float] = None,
    notional_eur: Optional[float] = None,
    is_stop_loss: bool = False,
    evaluate_intelligence: bool = True,
):
    """Return whether the order is profitable relative to its reference.

    Execution Pipeline:
    - Emergency Stop-Loss exits (is_stop_loss=True) bypass profit reference and are exempt from guards.
    - Stage 1 (Super-Matematică): Deterministic profit margin vs historical reference.
      If mathematically unprofitable, returns False immediately without evaluating
      AI/macro sentiment or sending false-positive push alerts.
    - Stage 2 (Intelligence, Macro & Sentiment): Evaluates Parabolic, Weibull, Gemini LLM,
      and Geopolitical shock shield only on mathematically cleared orders when
      evaluate_intelligence=True.
    """
    if is_stop_loss:
        return True

    # Stage 1: Deterministic Mathematical Reference Check
    if not check_math_profit_reference(
        provider, symbol, order_type, price, profit_percentage,
        window_ref=window_ref, regime_context=regime_context,
    ):
        return False

    if not evaluate_intelligence or qty is None or (isinstance(qty, (int, float)) and qty <= 0):
        return True

    # Stage 2: Intelligence, Macro & Sentiment Guards
    intel_ok, intel_reason, suggested_scale = check_intelligence_guards(
        provider, symbol, order_type, price, regime_context=regime_context, qty=qty, notional_eur=notional_eur,
        is_stop_loss=is_stop_loss,
    )
    if not intel_ok:
        print(f"[GUARD] {order_type} {symbol}: blocked by intelligence guard ({intel_reason})")
        return False

    if regime_context is not None and suggested_scale < 1.0:
        try:
            curr = getattr(regime_context, "suggested_scale", None)
            if curr is None or suggested_scale < curr:
                object.__setattr__(regime_context, "suggested_scale", float(suggested_scale))
                object.__setattr__(regime_context, "intelligence_scale_reason", intel_reason)
        except Exception:
            pass

    return True
