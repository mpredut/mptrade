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

_DEFAULT_REGIME_SERVICE = MarketRegimeService()
_MARGINS = None   # cache: {provider_lower: percentage, "default": 1.15}
_SHADOW_NOTIFY_COOLDOWN: Dict[Tuple[str, str, str], float] = {}


def _load_margins():
    """Read and cache `key = value` lines from order_guard.conf once.
    The configuration file is the SINGLE source of truth. The dictionary below is only
    a safety net for missing entries, such as a truncated file, and centralizes fallbacks
    so hard-coded shadow defaults are not scattered across functions. Change operational
    values in order_guard.conf, not here."""
    global _MARGINS
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
            if regime_context.is_valid_for(symbol=symbol, provider=provider, max_age_seconds=max_age, now=now):
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
        if not hasattr(regime_context, "is_valid_for") or regime_context.is_valid_for(symbol=symbol, provider=provider, max_age_seconds=max_age, now=now):
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
    recent = provider.get_orders(symbol, opp, window_s) or []
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
    recent = provider.get_orders(symbol, order_type, 86400) or []
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
    trades = provider.get_orders(symbol, order_type, safeback_sec) or []
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
    if _whale_collector is None:
        from intelligence.external.collectors.whale_positioning import WhalePositioningCollector
        _whale_collector = WhalePositioningCollector()
    return _whale_collector


def get_orderbook_collector():
    global _orderbook_collector
    if _orderbook_collector is None:
        from intelligence.external.collectors.orderbook_depth import OrderbookDepthCollector
        _orderbook_collector = OrderbookDepthCollector()
    return _orderbook_collector


def get_derivatives_collector():
    global _derivatives_collector
    if _derivatives_collector is None:
        from intelligence.external.collectors.derivatives_telemetry import DerivativesTelemetryCollector
        _derivatives_collector = DerivativesTelemetryCollector()
    return _derivatives_collector


def _read_cached_trend_duration(symbol: str) -> float:
    try:
        p = "cachedb/cache_price_long_trend.json"
        if not os.path.exists(p):
            return 0.0
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data.get("items", {}).get(symbol.upper(), [])
        if items and isinstance(items, list) and items[0]:
            return float(items[0].get("duration_seconds", 0.0) or 0.0)
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
    """Read rolling price history from local cache for parabolic surge evaluation."""
    try:
        p = "cachedb/cache_prices_multi.json"
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw_items = data.get("items", {}).get(symbol.upper(), [])
        if not raw_items:
            return None
        now_ts = time.time()
        cutoff_ts = now_ts - window_seconds
        history: List[Tuple[float, float]] = []
        for item in raw_items:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                ts_raw = float(item[0])
                ts_sec = ts_raw / 1000.0 if ts_raw > 1e11 else ts_raw
                if ts_sec >= cutoff_ts:
                    history.append((ts_sec, float(item[1])))
        return history if history else None
    except Exception:
        return None


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
    valid_context = False
    if regime_context is not None:
        max_age = _load_margins().get("regime_context_max_age_sec", 120.0)
        if hasattr(regime_context, "is_valid_for"):
            valid_context = regime_context.is_valid_for(
                symbol=symbol, provider=provider, max_age_seconds=max_age, now=now
            )
        else:
            valid_context = True

    if valid_context:
        cached_res = getattr(regime_context, "_intelligence_decision", None)
        if cached_res is not None:
            return cached_res

    res = _evaluate_intelligence_guards_raw(
        provider,
        symbol,
        order_type,
        price,
        price_history=price_history,
        trend_duration_seconds=trend_duration_seconds,
        regime_context=regime_context if valid_context else None,
        qty=qty,
        notional_eur=notional_eur,
        now=now,
    )

    if valid_context:
        try:
            object.__setattr__(regime_context, "_intelligence_decision", res)
        except Exception:
            pass

    return res


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
    now: Optional[float] = None,
) -> tuple[bool, str, float]:
    m = _load_margins()
    mode = str(m.get("intelligence_guards_mode", "shadow")).strip().lower()
    if mode in ("off", "0", "disabled"):
        return True, "intelligence_guards_off", 1.0

    side = (order_type or "").upper()
    if side != "BUY":
        return True, "not_buy_side", 1.0

    from intelligence.internal.guards.parabolic_guard import ParabolicSurgeGuard
    from intelligence.internal.guards.exhaustion_guard import WeibullExhaustionGuard
    from intelligence.internal.guards.guard_decision import BrakeAction

    surge_pct = float(m.get("parabolic_surge_pct", 15.0))
    pullback_pct = float(m.get("parabolic_pullback_pct", 2.0))
    exhaust_policy = str(m.get("weibull_exhaustion_policy", "downscale")).strip().lower()
    exhaust_scale = float(m.get("weibull_exhausted_scale", 0.25))

    p_guard = ParabolicSurgeGuard(surge_threshold_pct=surge_pct, pullback_required_pct=pullback_pct)
    e_guard = WeibullExhaustionGuard(policy=exhaust_policy, exhausted_scale=exhaust_scale)

    effective_scale = 1.0
    active_reason = "ok"

    # 1. Parabolic surge check (if price history is available or in local cache)
    history = price_history
    if history is None:
        history = _read_cached_price_history(symbol, window_seconds=7200.0)
    if history:
        p_dec = p_guard.check(symbol, side, price, price_history=history, now=now)
        if not p_dec.allowed:
            prefix = "[INTELLIGENCE_GUARD_SHADOW]" if mode == "shadow" else "[INTELLIGENCE_GUARD_ENFORCE]"
            print(f"{prefix} {side} {symbol} @ {price}: {p_dec.reason} (brake={p_dec.brake_action})")
            if mode == "enforce":
                return False, p_dec.reason, 0.0

    # 2. Weibull trend exhaustion check
    dur_sec = trend_duration_seconds
    if not dur_sec:
        if regime_context is not None:
            dur_sec = getattr(regime_context, "trend_duration_seconds", 0.0) or 0.0
        if not dur_sec:
            dur_sec = _resolve_trend_duration(symbol)

    if dur_sec and dur_sec > 0:
        e_dec = e_guard.check(symbol, side, trend_duration_seconds=dur_sec)
        if e_dec.brake_action != BrakeAction.NONE:
            prefix = "[INTELLIGENCE_GUARD_SHADOW]" if mode == "shadow" else "[INTELLIGENCE_GUARD_ENFORCE]"
            print(f"{prefix} {side} {symbol} @ {price}: {e_dec.reason} (brake={e_dec.brake_action}, suggested_scale={e_dec.suggested_scale})")
            if mode == "enforce":
                if not e_dec.allowed:
                    return False, e_dec.reason, 0.0
                effective_scale = min(effective_scale, e_dec.suggested_scale)
                active_reason = e_dec.reason

    shadow_notify = bool(int(float(m.get("shadow_notify", 1.0))))

    # 3. Google Gemini High-Stake Guard (> 1000 EUR purchases)
    gemini_mode = str(m.get("gemini_guard_mode", "shadow")).strip().lower()
    if gemini_mode not in ("off", "0", "disabled"):
        min_notional = float(m.get("gemini_min_notional_eur", 1000.0))
        computed_notional = notional_eur
        if computed_notional is None and qty is not None and price > 0:
            computed_notional = price * qty
        if computed_notional is not None and computed_notional >= min_notional:
            from intelligence.sentiment.guards.gemini_high_stake_guard import GeminiHighStakeGuard
            timeout_sec = float(m.get("gemini_timeout_sec", 12.0))
            fallback = str(m.get("gemini_fallback", "allow")).strip().lower()
            g_guard = GeminiHighStakeGuard(min_notional_eur=min_notional, timeout_sec=timeout_sec, fallback_action=fallback)
            g_dec = g_guard.check(symbol, side, price, qty if qty is not None else 1.0, notional_eur=computed_notional)
            if not g_dec.allowed or g_dec.brake_action == BrakeAction.DOWNSCALE_QTY:
                prefix = "[GEMINI_GUARD_SHADOW]" if gemini_mode == "shadow" else "[GEMINI_GUARD_ENFORCE]"
                print(f"{prefix} {side} {symbol} €{computed_notional:.2f}: {g_dec.reason} (brake={g_dec.brake_action}, suggested_scale={g_dec.suggested_scale})")
                if gemini_mode == "shadow" and shadow_notify:
                    cd_key = ("gemini", symbol, side)
                    now_ts = time.time()
                    if now_ts - _SHADOW_NOTIFY_COOLDOWN.get(cd_key, 0.0) >= 1800.0:
                        _SHADOW_NOTIFY_COOLDOWN[cd_key] = now_ts
                        try:
                            from notify_engine.alertnotifiers import notify
                            notify(
                                title=f"🛡 [GEMINI SHADOW VETO] Would Block {side} {symbol}",
                                body=f"High-stake order €{computed_notional:.2f} flagged: {g_dec.reason} (brake={g_dec.brake_action}, scale={g_dec.suggested_scale})",
                                source="order_guard",
                                symbol=symbol,
                            )
                        except Exception:
                            pass
                if gemini_mode == "enforce":
                    if not g_dec.allowed:
                        return False, g_dec.reason, 0.0
                    effective_scale = min(effective_scale, g_dec.suggested_scale)
                    active_reason = g_dec.reason

    # 4. Geopolitical & Energy Shock Guard (Black Swan Shield)
    geo_mode = str(m.get("geopolitical_guard_mode", "shadow")).strip().lower()
    if geo_mode not in ("off", "0", "disabled"):
        try:
            from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAnalyzer
            from intelligence.macro.geopolitical_guard import GeopoliticalShockGuard
            analyzer = GeopoliticalThreatAnalyzer()
            cached_geo = analyzer._cached_assessment
            if cached_geo is not None:
                geo_guard = GeopoliticalShockGuard()
                geo_dec = geo_guard.check(symbol, side, cached_geo)
                if not geo_dec.allowed or geo_dec.brake_action == BrakeAction.DOWNSCALE_QTY:
                    prefix = "[GEOPOLITICAL_GUARD_SHADOW]" if geo_mode == "shadow" else "[GEOPOLITICAL_GUARD_ENFORCE]"
                    print(f"{prefix} {side} {symbol}: {geo_dec.reason} (brake={geo_dec.brake_action}, suggested_scale={geo_dec.suggested_scale})")
                    if geo_mode == "shadow" and shadow_notify:
                        cd_key = ("geopolitical", symbol, side)
                        now_ts = time.time()
                        if now_ts - _SHADOW_NOTIFY_COOLDOWN.get(cd_key, 0.0) >= 1800.0:
                            _SHADOW_NOTIFY_COOLDOWN[cd_key] = now_ts
                            try:
                                from notify_engine.alertnotifiers import notify
                                notify(
                                    title=f"🛡 [MACRO SHADOW VETO] Would Block {side} {symbol}",
                                    body=f"Macro shock flagged: {geo_dec.reason} (threat={cached_geo.threat_level}, risk={cached_geo.risk_score:.2f})",
                                    source="order_guard",
                                    symbol=symbol,
                                )
                            except Exception:
                                pass
                    if geo_mode == "enforce":
                        if not geo_dec.allowed:
                            return False, geo_dec.reason, 0.0
                        effective_scale = min(effective_scale, geo_dec.suggested_scale)
                        active_reason = geo_dec.reason
        except Exception:
            pass

    # 5. Pillar 2: Whale Flow & Open Interest Divergence Guard
    whale_mode = str(m.get("whale_guard_mode", "shadow")).strip().lower()
    if whale_mode not in ("off", "0", "disabled"):
        try:
            from intelligence.external.guards.whale_divergence_guard import WhaleDivergenceGuard
            w_snap = get_whale_collector().fetch(symbol, allow_network=False)
            w_guard = WhaleDivergenceGuard()
            w_dec = w_guard.check(symbol, side, snapshot=w_snap)
            if not w_dec.allowed or w_dec.brake_action == BrakeAction.DEFER_WAIT:
                prefix = "[WHALE_GUARD_SHADOW]" if whale_mode == "shadow" else "[WHALE_GUARD_ENFORCE]"
                print(f"{prefix} {side} {symbol}: {w_dec.reason} (brake={w_dec.brake_action})")
                if whale_mode == "enforce":
                    if not w_dec.allowed:
                        return False, w_dec.reason, 0.0
        except Exception:
            pass

    # 6. Pillar 2: Orderbook Depth & Whale Limit Wall Guard
    ob_mode = str(m.get("orderbook_wall_guard_mode", "shadow")).strip().lower()
    if ob_mode not in ("off", "0", "disabled"):
        try:
            from intelligence.external.guards.orderbook_wall_guard import OrderbookWallGuard
            ob_snap = get_orderbook_collector().fetch(symbol, allow_network=False)
            min_imb = float(m.get("min_buy_imbalance", 0.25))
            wall_limit = float(m.get("whale_wall_usd_limit", 1_000_000.0))
            ob_guard = OrderbookWallGuard(
                min_buy_imbalance=min_imb,
                whale_wall_usd_limit=wall_limit,
            )
            ob_dec = ob_guard.check(symbol, side, snapshot=ob_snap)
            if not ob_dec.allowed or ob_dec.brake_action == BrakeAction.DEFER_WAIT:
                prefix = "[ORDERBOOK_GUARD_SHADOW]" if ob_mode == "shadow" else "[ORDERBOOK_GUARD_ENFORCE]"
                print(f"{prefix} {side} {symbol}: {ob_dec.reason} (brake={ob_dec.brake_action})")
                if ob_mode == "enforce":
                    if not ob_dec.allowed:
                        return False, ob_dec.reason, 0.0
        except Exception:
            pass

    # 7. Pillar 2: Perpetual Derivatives Funding Rate Crowding Guard
    funding_mode = str(m.get("funding_guard_mode", "shadow")).strip().lower()
    if funding_mode not in ("off", "0", "disabled"):
        try:
            from intelligence.external.guards.funding_crowding_guard import FundingCrowdingGuard
            f_snap = get_derivatives_collector().fetch(symbol, allow_network=False)
            f_max = float(m.get("funding_max_long_rate", 0.0005))
            f_policy = str(m.get("funding_crowding_policy", "downscale")).strip().lower()
            f_scale = float(m.get("funding_crowded_scale", 0.50))
            f_guard = FundingCrowdingGuard(
                max_long_funding_rate=f_max,
                crowding_policy=f_policy,
                crowded_scale=f_scale,
            )
            f_dec = f_guard.check(symbol, side, telemetry=f_snap)
            if not f_dec.allowed or f_dec.brake_action == BrakeAction.DOWNSCALE_QTY:
                prefix = "[FUNDING_GUARD_SHADOW]" if funding_mode == "shadow" else "[FUNDING_GUARD_ENFORCE]"
                print(f"{prefix} {side} {symbol}: {f_dec.reason} (brake={f_dec.brake_action}, suggested_scale={f_dec.suggested_scale})")
                if funding_mode == "enforce":
                    if not f_dec.allowed:
                        return False, f_dec.reason, 0.0
                    effective_scale = min(effective_scale, f_dec.suggested_scale)
                    active_reason = f_dec.reason
        except Exception:
            pass

    return True, active_reason if effective_scale < 1.0 else "ok", effective_scale


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
):
    """Return whether the order is profitable relative to its reference.
    Reference cascade: caller-provided window_ref first, otherwise
    provider.last_opposite_fill(symbol, order_type). A missing or non-positive reference
    allows placement because there is no prior transaction to compare."""
    # Staged / Shadow intelligence guard evaluation
    intel_ok, intel_reason, _ = check_intelligence_guards(
        provider, symbol, order_type, price, regime_context=regime_context, qty=qty, notional_eur=notional_eur
    )
    if not intel_ok:
        print(f"[GUARD] {order_type} {symbol}: blocked by intelligence guard ({intel_reason})")
        return False

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
                    symbol=symbol, provider=provider_name, max_age_seconds=max_age
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
            if window_ref is not None and window_ref > 0:
                diff = u.value_diff_to_percent(window_ref, price)
                if diff < profit_percentage:
                    # In dynamic mode, verify whether this reference actually sits within the dynamic window (dyn_window_s)
                    if hasattr(provider, "get_orders"):
                        recent = provider.get_orders(symbol, "SELL", dyn_window_s) or []
                        recent_prices = [float(o.get("price") or 0) for o in recent if float(o.get("price") or 0) > 0]
                        if not recent_prices:
                            print(f"[GUARD] BUY {symbol}: dynamic window ({dyn_hours:.1f}h, trend='{trend}') has no fills; "
                                  f"anchor {window_ref} from older period bypassed")
                            return True
                    elif trend == "bull" and not hasattr(provider, "get_orders"):
                        print(f"[GUARD] BUY {symbol}: dynamic mode bypassed historical sell reference "
                              f"during confirmed BULL trend; price {price} permitted")
                        return True
                    print(f"[GUARD] BUY {symbol}: dynamic mode ({dyn_hours:.1f}h window, trend='{trend}') "
                          f"found recent sell ref {window_ref}, price {price}, diff {diff:.2f}%, threshold {profit_percentage}%")
                    print(f"Percentage difference ({diff:.2f}%) below threshold {profit_percentage}%. "
                          f"The BUY order is BLOCKED.")
                    return False
                print(f"[GUARD] BUY {symbol}: dynamic mode ({dyn_hours:.1f}h window, trend='{trend}') "
                      f"recent sell ref {window_ref}, price {price}, diff {diff:.2f}% >= {profit_percentage}%. Permitted.")
                return True
            else:
                print(f"[GUARD] BUY {symbol}: dynamic mode ({dyn_hours:.1f}h window, trend='{trend}') "
                      f"has no recent sell reference; price {price} permitted")
                return True
    has_window = window_for(provider_name, symbol, order_type, regime_context=regime_context) > 0
    ref = window_ref if window_ref is not None else (
        None if has_window else (
            provider.last_opposite_fill(symbol, order_type) if hasattr(provider, "last_opposite_fill") else None
        )
    )
    if ref is None or ref <= 0:
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
