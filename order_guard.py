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
`order_guard.conf`. A missing file or required key aborts startup immediately."""
from functools import partial
import math
import statistics
from botcore import (
    _parse_bool_flag,
    clean_scale,
    exact_nonneg_int,
    normalize_side,
    obj_attr,
    require_int,
    require_nonneg_float,
    resolve_now_ts,
    resolve_order_notional,
    resolve_patched_attr,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
    value_diff_to_percent,
)
from typing import Optional, Callable, Any, Tuple
from cachepaths import (
    get_cachedb_dir,  # noqa: F401
    normalize_venue_name,
)
import guard_hierarchy as _gh
from guard_hierarchy import (
    GUARD_REGISTRY,
    _is_valid_regime_context,
    _load_margins,
    _validate_margins_cfg,
    _venue_cfg,
    is_category_bypassed,
    is_master_bypass,
    make_intel_cache_key,
    merge_guard_scale,
    normalize_bypassed_guards,
    normalize_intel_bypass_with_qlimit2_subs,
    preflight_intelligence_guard_entry,
    resolve_effective_bypass_guards,
    resolve_high_stake_min_notional_usd,
    set_regime_context_attrs,
    validate_guard_mode_cfg,
    validate_profit_guard_mode_cfg,
)
import external_order_guard as _ext_order_guard
from intelligence.internal.state.survival import (
    _read_cached_price_history,
    _resolve_trend_direction,
    _resolve_trend_duration,
    get_weight_for_cash_permission_at_quant_time,
    resolve_effective_trend_duration,
)
import llm_order_guard as _llm_order_guard
from market_regime import MarketRegimeService, MarketRegimeDecision, MarketRegimeContext
import providers.market_api as _market_api_mod
import providers.order_id_context as _order_id_ctx
from providers.order_ohlc_utils import (
    extract_order_price,
    extract_order_qty,
    extract_timestamp_ms,
    extract_timestamp_s,
)
from providers.strategy_executor import ProviderError, SubmissionRefused
import qlimit_order_guard as _qlimit_order_guard

_DEFAULT_REGIME_SERVICE = MarketRegimeService()
_MARGINS = _gh._MARGINS
_MARGINS_FILE_MTIME: float = _gh._MARGINS_FILE_MTIME
_MARGINS_LAST_CHECK: float = _gh._MARGINS_LAST_CHECK
_load_margins()


_PROCESS_TO_MODULE = {
    "AG": "assetguardian",
    "MT": "monitortrades",
    "TA": "tradeall",
    "RT": "rtrade",
}
_KNOWN_GUARD_MODULES = frozenset(_PROCESS_TO_MODULE.values())


def resolve_caller_module(
    module: Optional[str] = None,
    *,
    prefix: Optional[str] = None,
    client_order_id: Optional[str] = None,
) -> Optional[str]:
    """Resolve canonical module identity from explicit module, order lineage prefix, or process name."""
    if module and not isinstance(module, bool):
        m_raw = str(module).strip()
        if m_raw:
            m_up = m_raw.upper()
            if m_up in _PROCESS_TO_MODULE:
                return _PROCESS_TO_MODULE[m_up]
            m_low = m_raw.lower()
            if m_low == "asset_guardian":
                return "assetguardian"
            return m_low
    if (client_order_id and not isinstance(client_order_id, bool)) or (prefix and not isinstance(prefix, bool)):
        try:
            for candidate in (client_order_id, prefix):
                if not candidate or isinstance(candidate, bool):
                    continue
                owner = _order_id_ctx.resolve_order_owner(candidate)
                if owner in _KNOWN_GUARD_MODULES:
                    return owner
        except Exception:
            pass
    try:
        return _PROCESS_TO_MODULE.get(_order_id_ctx.get_process_name())
    except Exception:
        return None


def _require_buy_sell_side(order_type: str) -> str:
    side = normalize_side(order_type)
    if side not in ("BUY", "SELL"):
        raise ValueError(f"Invalid order_type {order_type!r}; expected 'BUY' or 'SELL'")
    return side


def opposite_side(order_type: str) -> str:
    """Return the opposite order side ('SELL' for 'BUY', 'BUY' for 'SELL')."""
    return "SELL" if _require_buy_sell_side(order_type) == "BUY" else "BUY"


def profit_diff_pct(order_type: str, price: float, ref_price: float) -> float:
    """Compute signed favorable profit percentage of `price` relative to `ref_price` symmetrically."""
    side = _require_buy_sell_side(order_type)
    return value_diff_to_percent(ref_price, price) if side == "BUY" else value_diff_to_percent(price, ref_price)


def select_extreme_reference(prices, order_type: str, *, strict: bool = True):
    """Select the directional extreme reference symmetrically for BUY and SELL."""
    side = _require_buy_sell_side(order_type)
    if prices is None or isinstance(prices, (bool, str, bytes)):
        return None
    try:
        clean_prices = [pf for p in prices if (pf := safe_pos_float(p, default=None)) is not None]
    except TypeError:
        return None
    if not clean_prices:
        return None
    if _parse_bool_flag(strict, True):
        return min(clean_prices) if side == "BUY" else max(clean_prices)
    return max(clean_prices) if side == "BUY" else min(clean_prices)


def get_profit_guard_mode_cfg(
    provider_name,
    order_type: str,
    *,
    module: Optional[str] = None,
    reference_mode: Optional[str] = None,
) -> str:
    """Read the BUY or SELL profit guard difficulty mode strictly from order_guard.conf.

    Fails fast (ValueError / KeyError) if any key or value is missing or invalid.
    """
    if reference_mode is not None:
        return validate_profit_guard_mode_cfg(reference_mode)

    side_u = normalize_side(order_type)
    if side_u not in ("BUY", "SELL"):
        raise ValueError(f"Invalid order_type {order_type!r}; expected 'BUY' or 'SELL'")
    side_l = side_u.lower()

    venue = normalize_venue_name(provider_name, default="").strip().lower()
    if not venue:
        raise ValueError(f"Missing provider_name for get_profit_guard_mode_cfg({provider_name!r}, {order_type!r})")

    m = _load_margins()
    mod = resolve_caller_module(module)

    if mod:
        mod_key = f"{mod}_{side_l}_reference"
        if mod_key not in m:
            raise KeyError(f"Missing required module config key {mod_key!r} in order_guard.conf")
        mod_val = str(m[mod_key]).strip().lower()
        if mod_val != "venue":
            return validate_profit_guard_mode_cfg(mod_val)

    return validate_profit_guard_mode_cfg(
        _venue_cfg(venue, f"_{side_l}_reference", margins=m, required=True)
    )


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
            snap = _market_api_mod._runtime_market_regime_snapshot(symbol, now=now)
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
            target_provider = _market_api_mod.api.provider_by_name(provider) or provider
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
    now_ts = resolve_now_ts(now, allow_zero=True)
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
    dur_sec = resolve_effective_trend_duration(symbol, trend_duration_seconds)
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
    if _is_valid_regime_context(regime_context, symbol, provider, now=now):
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


def get_margin_cfg(provider_name):
    """Return the venue's configured minimum profit percentage from order_guard.conf."""
    return _venue_cfg(provider_name)


def _trend_matches_sides(side: str, trend: Any, buy_trend: str, sell_trend: str) -> bool:
    side_u = normalize_side(side)
    trend_l = trend.strip().lower() if isinstance(trend, str) else ""
    return (side_u == "BUY" and trend_l == buy_trend) or (side_u == "SELL" and trend_l == sell_trend)


_is_favorable_trend = partial(_trend_matches_sides, buy_trend="bull", sell_trend="bear")
is_adverse_trend = partial(_trend_matches_sides, buy_trend="bear", sell_trend="bull")


def get_dynamic_window_sec_cfg(
    symbol=None,
    provider=None,
    order_type: str = "BUY",
    regime_service=None,
    now=None,
    *,
    resolved_trend=None,
    regime_context=None,
    snapshot_resolver=None,
    provider_resolver=None,
) -> float:
    """Calculate the dynamic lookback window (in seconds) symmetrically for BUY and SELL:
    - Favorable trend (BUY in bull / SELL in bear): 4h - 12h (configured via dynamic_window_favorable_h)
    - Adverse trend (BUY in bear / SELL in bull): 48h - 168h (configured via dynamic_window_adverse_h)
    - Flat / unknown trend: 12h - 48h (configured via dynamic_window_flat_h)
    """
    side = "BUY" if (order_type is None or order_type == "") else normalize_side(order_type)
    if side not in ("BUY", "SELL"):
        raise ValueError(f"Invalid order_type {order_type!r}; expected 'BUY' or 'SELL'")

    m = _load_margins()
    favorable_h = safe_pos_float(m.get("dynamic_window_favorable_h"), 2.0)
    flat_h = safe_pos_float(m.get("dynamic_window_flat_h"), 2.0)
    adverse_h = safe_pos_float(m.get("dynamic_window_adverse_h"), 2.0)

    trend = resolved_trend
    if trend is None and _is_valid_regime_context(regime_context, symbol, provider, margins=m, now=now):
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
    if _is_favorable_trend(side, trend):
        hours = max(4.0, min(12.0, favorable_h))
    elif is_adverse_trend(side, trend):
        hours = max(48.0, min(168.0, adverse_h))
    else:  # flat, sideways, chop, or unknown
        hours = max(12.0, min(48.0, flat_h))
    return hours * 3600.0


def get_window_cfg(
    provider_name,
    symbol=None,
    order_type=None,
    *,
    regime_context=None,
    module=None,
    reference_mode=None,
) -> float:
    """Return the configured reference window in seconds from order_guard.conf symmetrically
    for the given venue and side.

    When dynamic windowing is active (mode == 'dynamic'), returns get_dynamic_window_sec_cfg(...)
    scaling lookback from 4h-12h in favorable trend to 48h-168h in adverse trend.
    Otherwise returns `<venue>_window_h` (or default_window_h) in seconds.
    """
    name = normalize_venue_name(provider_name, default="")
    side = normalize_side(order_type)

    if side in ("BUY", "SELL"):
        mode = get_profit_guard_mode_cfg(
            name, side, module=module, reference_mode=reference_mode
        )
        if mode == "off":
            return 0.0
        if mode == "dynamic":
            return get_dynamic_window_sec_cfg(
                symbol,
                provider=name,
                order_type=side,
                regime_context=regime_context,
            )

    return require_nonneg_float(_venue_cfg(name, "_window_h"), name="window_h") * 3600.0


def _order_ts_qty_price(o):
    if not isinstance(o, dict):
        return 0.0, 0.0, 0.0
    return (
        extract_timestamp_s(o, default=0.0) or 0.0,
        extract_order_qty(o, default=0.0) or 0.0,
        extract_order_price(o, default=0.0) or 0.0,
    )


def _aggregate_reference_prices(active_lots, valid_opp, side: str, mode: str):
    """Aggregate active lots into a reference price symmetrically for BUY and SELL."""
    active_prices = []
    valid_weighted = []
    for item in (active_lots or []):
        try:
            q_raw, px_raw = item
        except (TypeError, ValueError):
            continue
        px_f = safe_pos_float(px_raw, 0.0)
        if px_f > 0.0:
            active_prices.append(px_f)
            q_f = safe_pos_float(q_raw, 0.0)
            if q_f > 0.0:
                valid_weighted.append((q_f, px_f))
    if not active_prices:
        return None
    strict_ref = select_extreme_reference(active_prices, side, strict=True)
    if mode in ("on", "strict", "dynamic"):
        return strict_ref
    if mode == "off":
        return None
    weighted_qty = sum(q for q, _ in valid_weighted)
    if weighted_qty > 0:
        avg_ref = sum(q * px for q, px in valid_weighted) / weighted_qty
    else:
        avg_ref = statistics.fmean(active_prices)
    finite_ts_opp = []
    for item in (valid_opp or []):
        try:
            ts_raw, _, px_raw = item
        except (TypeError, ValueError):
            continue
        ts_f = safe_nonneg_float(ts_raw, -1.0)
        px_f = safe_pos_float(px_raw, 0.0)
        if ts_f >= 0.0 and px_f > 0.0:
            finite_ts_opp.append((ts_f, px_f))
    if finite_ts_opp:
        last_ref = max(finite_ts_opp, key=lambda x: x[0])[1]
    else:
        last_ref = active_prices[-1]
    if mode == "average":
        return avg_ref
    if mode == "last":
        return last_ref
    if mode == "hybrid":
        return select_extreme_reference([avg_ref, last_ref], side, strict=False)
    return strict_ref


def active_cycle_window_ref(opp_orders, same_orders=None, order_type="SELL", *, reference_mode=None):
    """Compute the window reference price from opposite-side fills, excluding earlier
    cycles that were already closed by subsequent same-side fills within the window."""
    if not opp_orders:
        return None
    side = normalize_side(order_type)
    if side not in ("BUY", "SELL"):
        return None
    mode = validate_profit_guard_mode_cfg(reference_mode or "strict")
    if mode == "off":
        return None
    valid_opp = []
    for o in opp_orders:
        ts, qty, px = _order_ts_qty_price(o)
        if px > 0:
            valid_opp.append((ts, qty, px))
    if not valid_opp:
        return None
    default_lots = [(qty, px) for _, qty, px in valid_opp]
    default_ref = _aggregate_reference_prices(default_lots, valid_opp, side, mode)
    if len(valid_opp) <= 1 or not same_orders:
        return default_ref

    valid_same = []
    for o in same_orders:
        ts, qty, px = _order_ts_qty_price(o)
        if ts > 0 and qty > 0:
            valid_same.append((ts, qty, px))
    if not valid_same or not all(ts > 0 and qty > 0 for ts, qty, _ in valid_opp):
        return default_ref

    events = (
        [(ts, 0, qty, px) for ts, qty, px in valid_opp]
        + [(ts, 1, qty, px) for ts, qty, px in valid_same]
    )
    events.sort(key=lambda x: (x[0], x[1]))

    lots = []  # list of [remaining_qty, initial_qty, price, ts]
    for ts, is_same, qty, px in events:
        if is_same == 0:
            lots.append([qty, qty, px, ts])
        else:
            rem_to_consume = qty
            while lots and rem_to_consume > 1e-12:
                rem_lot, init_lot, _lot_px, _lot_ts = lots[-1]
                if rem_lot <= rem_to_consume + 1e-12:
                    rem_to_consume -= rem_lot
                    lots.pop()
                else:
                    new_rem = rem_lot - rem_to_consume
                    rem_to_consume = 0.0
                    if new_rem <= init_lot * 0.02 + 1e-12:
                        lots.pop()
                    else:
                        lots[-1][0] = new_rem

    active_lots = [(lot[0], lot[2]) for lot in lots if lot[0] > lot[1] * 0.02 + 1e-12 and lot[2] > 0]
    active_opp = [(lot[3], lot[0], lot[2]) for lot in lots if lot[0] > lot[1] * 0.02 + 1e-12 and lot[2] > 0]
    if not active_lots:
        return default_ref
    return _aggregate_reference_prices(active_lots, active_opp, side, mode)


def _resolve_window_ref_from_orders(
    provider,
    symbol: str,
    side: str,
    window_s: float,
    recent: list,
    eff_mode: str = "strict",
):
    """Compute window reference price from `recent` opposite-side orders and optional same-side orders."""
    prices = [
        px
        for o in recent
        if (px := extract_order_price(o, default=0.0)) > 0.0
    ]
    if not prices:
        return None
    if len(recent) > 1 and all(
        isinstance(o, dict) and (extract_timestamp_ms(o, default=0) or 0) > 0
        for o in recent
    ):
        try:
            same_orders = provider.get_orders(symbol, side, window_s) or []
        except Exception:
            same_orders = []
        if same_orders or eff_mode in ("average", "last", "hybrid"):
            active_ref = active_cycle_window_ref(recent, same_orders, side, reference_mode=eff_mode)
            if active_ref is not None and active_ref > 0:
                return active_ref
    if eff_mode in ("average", "last", "hybrid"):
        return active_cycle_window_ref(recent, None, side, reference_mode=eff_mode)
    return select_extreme_reference(prices, side, strict=True)


def window_reference(
    provider,
    symbol,
    order_type,
    window_s,
    *,
    reference_mode: Optional[str] = None,
    module: Optional[str] = None,
):
    """Return a time-windowed reference from opposite-side orders/fills symmetrically
    for BUY and SELL according to the resolved difficulty mode ('on'/'strict', 'average',
    'last', 'hybrid', 'dynamic', or 'off').
    Return None for an empty or disabled window. Read errors propagate so the caller fails closed."""
    window_s_f = safe_pos_float(window_s, default=None)
    if window_s_f is None:
        return None
    side = normalize_side(order_type)
    eff_mode = get_profit_guard_mode_cfg(
        provider, side, module=module, reference_mode=reference_mode
    )
    if eff_mode == "off":
        return None
    opp = opposite_side(side)
    recent = provider.get_orders(symbol, opp, window_s_f)
    if recent is None:
        raise ProviderError(f"window_reference({symbol}): failed to read order history")
    return _resolve_window_ref_from_orders(provider, symbol, side, window_s_f, recent, eff_mode=eff_mode)


def get_chop_weight_cfg(provider_name):
    """Return the venue's conservative chop fallback weight from order_guard.conf."""
    raw = _venue_cfg(provider_name, "_chop_weight")
    val = safe_pos_float(raw, default=None)
    if val is None:
        raise ValueError(f"Invalid chop_weight in order_guard.conf: {raw!r}")
    return val


def resolve_trade_weight(symbol, order_type, provider_name=None, *, default_weight=None, raise_on_error=False):
    """Resolve trade weight using survival Gaussian curve, proxy, or conservative fallback.

    1. Gaussian weight for symbol when long trend exists.
    2. Proxy Gaussian weight (e.g. BTCUSDC) when symbol has no trend.
    3. Conservative chop fallback (default 0.03 or venue-configured chop weight).
    """

    def _gauss(sym):
        try:
            weight_fn = resolve_patched_attr(
                "get_weight_for_cash_permission_at_quant_time",
                get_weight_for_cash_permission_at_quant_time,
                modules=("priceAnalysis", "intelligence.internal.state.survival"),
                require_callable=True,
            )
            val = weight_fn(sym, order_type)
            fval = safe_pos_float(val, default=None)
            if fval is not None and fval <= 1.0:
                return fval
            if val is not None:
                if raise_on_error and sym == symbol:
                    raise SubmissionRefused("invalid_weight_policy_weight")
                return None
            return None
        except Exception as e:
            if raise_on_error and sym == symbol:
                raise
            print(f"[WEIGHT] {sym}: cannot compute the gauss value ({e})")
            return None

    weight = _gauss(symbol)
    if weight is None:
        proxy = _venue_cfg(provider_name, "_weight_proxy")
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
            default_weight = get_chop_weight_cfg(provider_name)
        weight = default_weight
        print(f"[WEIGHT] {symbol}: no trend or proxy -> fallback weight={weight}")

    return safe_pos_float(weight, default=0.0) or 0.0


def _clean_required_qty_or_inf(required_qty) -> float:
    """Parse required_qty (>0), preserving +inf for auto-quantity while rejecting None/bool/NaN/<=0."""
    if isinstance(required_qty, (int, float)) and not isinstance(required_qty, bool) and math.isinf(required_qty):
        return math.inf if required_qty > 0 else 0.0
    if isinstance(required_qty, str) and required_qty.strip().lower() in ("inf", "+inf", "infinity", "+infinity"):
        return math.inf
    return safe_pos_float(required_qty, 0.0)


def compute_weight_capped_qty(weight, price, required_qty, available_qty, traded_24h_value):
    """Cap per-order quantity using Gaussian/chop weight and 24h traded quote reference."""
    w = min(1.0, safe_pos_float(weight, 0.0))
    px = safe_pos_float(price, 0.0)
    req = _clean_required_qty_or_inf(required_qty)
    if req <= 0.0 or w <= 0.0 or px <= 0.0:
        return 0.0
    avail = safe_pos_float(available_qty, 0.0)
    traded = safe_pos_float(traded_24h_value, 0.0)

    total_ref = traded + avail * px
    max_trade_value = total_ref * w
    remaining_value = max(0.0, max_trade_value - traded)
    remaining_qty = remaining_value / px if px > 0 else 0.0
    return min(req, remaining_qty)


def weight_limit(provider, symbol, order_type, price, required_qty, *, available_qty):
    """Cap per-order quantity using the Gaussian curve, the platform-agnostic equivalent
    of bapi.apply_weight_limit. Allocate tradable value by trend position so the whole
    amount is not bought or sold at once. priceAnalysis supplies the Gaussian weight;
    provider.get_orders supplies 24-hour traded value. The shared quantity decision must
    supply the side-aware balance, which this guard does not re-read. Return the smaller
    of requested and permitted quantity. Errors propagate so the caller fails closed."""
    side = _require_buy_sell_side(order_type)
    provider_name = normalize_venue_name(provider, default="")
    weight = resolve_trade_weight(symbol, side, provider_name)
    recent = provider.get_orders(symbol, side, 86400)
    if recent is None:
        raise ProviderError(f"weight_limit({symbol}): order history unavailable")
    traded_value = sum(
        qty * px
        for o in recent
        for _ts, qty, px in (_order_ts_qty_price(o),)
    )
    available = safe_pos_float(available_qty, 0.0)
    px_clean = safe_pos_float(price, 0.0)
    req_clean = _clean_required_qty_or_inf(required_qty)
    adjusted = compute_weight_capped_qty(weight, px_clean, req_clean, available, traded_value)
    total_ref = traded_value + available * px_clean
    max_trade_value = total_ref * weight
    remaining_value = max(0.0, max_trade_value - traded_value)
    print(f"[WEIGHT] {side} {symbol}: weight={weight} traded24h={traded_value:.2f} "
          f"avail={available:.6f} max={max_trade_value:.2f} remaining={remaining_value:.2f} "
          f"requested={req_clean:.6f} -> {adjusted:.6f}")
    return adjusted


DAILY_LIMIT_WINDOW_SEC = 86400.0


def daily_limit_guard(provider, symbol, order_type, max_daily_trades=None, *, bypass_guards=None):
    """Enforce GO1 symmetrically per symbol and side:
    maximum filled orders per 24 hours (86400s).

    Return (True, None) or (False, "daily_limit" | "history_unavailable").
    """
    if bypass_guards:
        bypassed = normalize_bypassed_guards(bypass_guards=bypass_guards)
        if GUARD_REGISTRY["GO1"].is_bypassed(bypassed):
            return True, None
    name = normalize_venue_name(provider, default="")
    order_type = normalize_side(order_type)
    if order_type not in ("BUY", "SELL"):
        return False, "invalid_side"
    resolved_max_trades = exact_nonneg_int(max_daily_trades, default=None)
    max_daily_trades = (
        resolved_max_trades
        if resolved_max_trades is not None
        else require_int(_venue_cfg(name, "_max_daily_trades"), name="max_daily_trades", min_value=0)
    )
    def _block_history_unavailable(detail: str) -> Tuple[bool, str]:
        print(f"[DAILY-LIMIT] {order_type} {symbol}: {detail} -> BLOCKED")
        return False, "history_unavailable"

    try:
        trades = provider.get_orders(symbol, order_type, DAILY_LIMIT_WINDOW_SEC)
    except ProviderError as exc:
        return _block_history_unavailable(f"history unavailable ({exc})")
    except Exception as exc:
        return _block_history_unavailable(f"history read failed ({exc})")
    if trades is None:
        return _block_history_unavailable("history unavailable")
    if len(trades) >= max_daily_trades:
        print(f"[DAILY-LIMIT] {order_type} {symbol}: {len(trades)} trades in 24h, "
              f"cap of {max_daily_trades}/24h -> BLOCKED")
        return False, "daily_limit"
    return True, None


def _record_context_suggested_scale(
    regime_context: Any,
    side: str,
    scale: float,
    reason: str,
    *,
    reason_attr: str = "intelligence_reason",
    only_if_smaller: bool = False,
) -> None:
    """Update `_suggested_scale_by_side` and (for BUY) `suggested_scale` on `regime_context`."""
    if regime_context is None:
        return
    is_dict = isinstance(regime_context, dict)
    try:
        scales_by_side = obj_attr(regime_context, "_suggested_scale_by_side")
        if not isinstance(scales_by_side, dict):
            scales_by_side = {}
        curr_side = clean_scale(scales_by_side.get(side), default=None, allow_percent=True)
        if not only_if_smaller or curr_side is None or scale < curr_side:
            scales_by_side[side] = scale
        if is_dict:
            regime_context["_suggested_scale_by_side"] = scales_by_side
        else:
            object.__setattr__(regime_context, "_suggested_scale_by_side", scales_by_side)
        if side == "BUY":
            curr = clean_scale(obj_attr(regime_context, "suggested_scale"), default=None, allow_percent=True)
            if not only_if_smaller or curr is None or scale < curr:
                if is_dict:
                    regime_context["suggested_scale"] = scale
                    regime_context[reason_attr] = reason
                else:
                    object.__setattr__(regime_context, "suggested_scale", scale)
                    object.__setattr__(regime_context, reason_attr, reason)
    except Exception:
        pass


def _update_context_intel_cache(
    regime_context: Any,
    cache_key: Any,
    price: Any,
    notional: Any,
    *,
    dur_sec: Optional[float] = None,
    decision: Any = None,
) -> None:
    if regime_context is None:
        return
    try:
        decisions_map = getattr(regime_context, "_intelligence_decisions_by_key", None)
        if not isinstance(decisions_map, dict):
            decisions_map = {}
        entry = dict(decisions_map.get(cache_key) or {})
        entry["price"] = price
        if decision is not None:
            entry["decision"] = decision
            entry["notional"] = notional
            if dur_sec is not None:
                entry["trend_duration_seconds"] = dur_sec
            decisions_map[cache_key] = entry
            set_regime_context_attrs(
                regime_context,
                _intelligence_decisions_by_key=decisions_map,
                _intelligence_decision_key=cache_key,
                _intelligence_decision=decision,
                _intelligence_notional=notional,
                **(
                    {"_intelligence_trend_duration_seconds": dur_sec}
                    if dur_sec is not None
                    else {}
                ),
            )
        elif cache_key in decisions_map:
            if notional is not None:
                entry["notional"] = notional
            decisions_map[cache_key] = entry
        object.__setattr__(regime_context, "_intelligence_price", price)
        if decision is None and notional is not None:
            object.__setattr__(regime_context, "_intelligence_notional", notional)
    except Exception:
        pass


def _preflight_intelligence_guards_with_qlimit2(
    order_type: Any,
    bypass_guards: Any = None,
) -> tuple[Optional[tuple[bool, str, float]], str, dict[str, Any], set[str], bool, bool]:
    bypassed, parabolic_only_bypass, weibull_only_bypass = normalize_intel_bypass_with_qlimit2_subs(bypass_guards)
    early, side, m, bypassed = preflight_intelligence_guard_entry(
        order_type,
        load_margins_fn=_load_margins,
        pre_normalized_bypassed=bypassed,
        require_master_key=True,
        check_master_before_side=True,
    )
    return early, side, m, bypassed, parabolic_only_bypass, weibull_only_bypass


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
    bypass_guards: Optional[set[str]] = None,
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
    early, side, m, bypassed, parabolic_only_bypass, weibull_only_bypass = _preflight_intelligence_guards_with_qlimit2(
        order_type, bypass_guards
    )
    if early is not None:
        return early
    valid_context = _is_valid_regime_context(regime_context, symbol, provider, margins=m, now=now)

    px_clean = safe_pos_float(price, 0.0)
    computed_notional = resolve_order_notional(notional_eur, px_clean, qty, default=None)
    explicit_dur_sec = safe_pos_float(trend_duration_seconds, 0.0)

    # Check if expensive static analysis is already cached on regime_context for this (side, intel_bypass) key
    cache_key = make_intel_cache_key(side, bypassed)
    legacy_cache_key = (side, frozenset(bypassed))
    decisions_map = getattr(regime_context, "_intelligence_decisions_by_key", None) if valid_context else None
    if isinstance(decisions_map, dict) and (cache_key in decisions_map or legacy_cache_key in decisions_map):
        raw_entry = decisions_map.get(cache_key)
        if not isinstance(raw_entry, dict):
            raw_entry = decisions_map.get(legacy_cache_key)
        cached_entry = raw_entry if isinstance(raw_entry, dict) else {}
        cached_res = cached_entry.get("decision")
        cached_price = cached_entry.get("price")
        cached_notional = cached_entry.get("notional")
        cached_dur = safe_pos_float(cached_entry.get("trend_duration_seconds"), 0.0)
    elif valid_context and getattr(regime_context, "_intelligence_decision_key", cache_key) in (cache_key, legacy_cache_key):
        cached_res = getattr(regime_context, "_intelligence_decision", None)
        cached_price = getattr(regime_context, "_intelligence_price", None)
        cached_notional = getattr(regime_context, "_intelligence_notional", None)
        cached_dur = safe_pos_float(getattr(regime_context, "_intelligence_trend_duration_seconds", 0.0), 0.0)
    else:
        cached_res = cached_price = cached_notional = None
        cached_dur = 0.0

    if (
        cached_res is not None
        and (
            (
                not cached_res[0]
                and side in ("BUY", "SELL")
                and ("parabolic_surge" in str(cached_res[1]) or "parabolic_dump" in str(cached_res[1]))
                and (price != cached_price or now is not None)
            )
            or (
                ("weibull_exhaustion" in str(cached_res[1]) or "trend_exhausted" in str(cached_res[1]))
                and explicit_dur_sec != cached_dur
            )
        )
    ):
        cached_res = None

    was_cached = cached_res is not None
    if cached_res is None:
        raw_bypass = (bypassed | {"gllm1"}) if valid_context else bypassed
        cached_res = _evaluate_intelligence_guards_raw(
            provider,
            symbol,
            side,
            price,
            price_history=price_history,
            trend_duration_seconds=explicit_dur_sec,
            regime_context=regime_context if valid_context else None,
            qty=qty,
            notional_eur=computed_notional,
            now=now,
            bypass_guards=raw_bypass,
        )
        if valid_context:
            _update_context_intel_cache(
                regime_context, cache_key, price, computed_notional, dur_sec=explicit_dur_sec, decision=cached_res
            )

    static_allowed, static_reason, static_scale = cached_res
    if not static_allowed:
        # Rejected by baseline or macro guard (e.g. Black Swan, Whale divergence, Ask Wall)
        return False, static_reason, 0.0

    if not valid_context:
        return cached_res

    effective_scale = static_scale
    active_reason = static_reason

    # 1. Dynamic G_QLIMIT2 re-evaluation (ParabolicSurgeGuard + WeibullExhaustionGuard) when cached inputs changed
    qlimit2_inputs_changed = (
        price != cached_price
        or explicit_dur_sec != cached_dur
        or price_history is not None
        or now is not None
    )
    if (
        was_cached
        and qlimit2_inputs_changed
        and not (GUARD_REGISTRY["G_QLIMIT2"].is_bypassed(bypassed) and not parabolic_only_bypass and not weibull_only_bypass)
    ):
        q_ok, q_reason, q_scale = _qlimit_order_guard.check_qlimit2_order_guards(
            symbol=symbol,
            side=side,
            price=px_clean,
            price_history=price_history,
            trend_duration_seconds=explicit_dur_sec,
            regime_context=regime_context,
            margins=m,
            now=now,
            bypass_guards=bypassed,
        )
        if not q_ok:
            return False, q_reason, 0.0
        effective_scale, active_reason = merge_guard_scale(effective_scale, active_reason, q_scale, q_reason)

    # 2. Dynamic LLM Pre-Trade Guard (GLLM1) if notional crossed the threshold
    dur_sec = resolve_effective_trend_duration(symbol, explicit_dur_sec, regime_context)

    hs_ok, active_reason, effective_scale = _llm_order_guard.evaluate_and_merge_high_stake_guard(
        effective_scale,
        active_reason,
        symbol=symbol,
        side=side,
        price=price,
        qty=qty,
        computed_notional=computed_notional,
        regime_context=regime_context,
        valid_context=valid_context,
        cache_key=cache_key,
        margins=m,
        bypassed=bypassed,
        now=now,
        dur_sec=dur_sec,
        fallback_notional=cached_notional,
        notify_on_eval=True,
    )
    if not hs_ok:
        return False, active_reason, 0.0

    final_res = (True, active_reason if effective_scale < 1.0 else "ok", effective_scale)
    if valid_context:
        _update_context_intel_cache(regime_context, cache_key, price, computed_notional)
        _record_context_suggested_scale(regime_context, side, effective_scale, active_reason)
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
    now: Optional[float] = None,
    bypass_guards: Optional[set[str]] = None,
) -> tuple[bool, str, float]:
    if (early := (pre := _preflight_intelligence_guards_with_qlimit2(order_type, bypass_guards))[0]) is not None:
        return early
    _, side, m, bypassed, parabolic_only_bypass, weibull_only_bypass = pre

    if any(fake in (symbol or "").upper() for fake in ("ZZZFAKE", "FAKEUSD", "TESTPAIR", "TSTX", "FAKE_VENUE")):
        return True, "test_symbol_exempt", 1.0

    computed_notional = resolve_order_notional(notional_eur, price, qty, default=None)
    effective_scale, active_reason = 1.0, "ok"

    # 1. Internal Quantitative Mathematical Guards (G_QLIMIT2: Parabolic Surge, Weibull Exhaustion)
    if not (GUARD_REGISTRY["G_QLIMIT2"].is_bypassed(bypassed) and not parabolic_only_bypass and not weibull_only_bypass):
        int_ok, int_reason, int_scale = _qlimit_order_guard.check_qlimit2_order_guards(
            symbol=symbol,
            side=side,
            price=price,
            price_history=price_history,
            trend_duration_seconds=trend_duration_seconds,
            regime_context=regime_context,
            margins=m,
            now=now,
            bypass_guards=bypassed,
        )
        if not int_ok:
            return False, int_reason, 0.0
        effective_scale, active_reason = merge_guard_scale(effective_scale, active_reason, int_scale, int_reason)

    # 2. External Microstructure & Derivatives Guards (GI_OBOOK, GI_WHALE, GI_FUNDING)
    if not is_category_bypassed("intelligence", bypassed):
        ext_ok, ext_reason, ext_scale = _ext_order_guard.check_external_order_guards(
            symbol=symbol,
            side=side,
            price=price,
            margins=m,
            now=now,
            bypass_guards=bypassed,
        )
        if not ext_ok:
            return False, ext_reason, 0.0
        effective_scale, active_reason = merge_guard_scale(effective_scale, active_reason, ext_scale, ext_reason)

    # 3. LLM Guards (GLLM1 High-Stake + GLLM2 Sentiment + GLLM3 Geopolitical)
    if not is_category_bypassed("llm", bypassed):
        intel_ok, intel_reason, intel_scale = _llm_order_guard.check_llm_order_guards(
            provider=provider,
            symbol=symbol,
            order_type=side,
            price=price,
            qty=qty,
            notional_eur=computed_notional,
            regime_context=regime_context,
            margins=m,
            now=now,
            bypass_guards=bypassed,
        )
        if not intel_ok:
            return False, intel_reason, 0.0
        effective_scale, active_reason = merge_guard_scale(effective_scale, active_reason, intel_scale, intel_reason)

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
    reference_mode: Optional[str] = None,
    module: Optional[str] = None,
) -> bool:
    """Evaluate deterministic price reference and margin percentage thresholds symmetrically
    for BUY and SELL across all difficulty tiers ('off', 'hybrid', 'last', 'average', 'dynamic', 'on').

    Pure mathematical calculation (0 tokens, deterministic, no AI or macro push alerts).
    """
    side = normalize_side(order_type)
    opp_side = opposite_side(order_type)
    provider_name = normalize_venue_name(provider, default="")
    mode = get_profit_guard_mode_cfg(
        provider_name, side, module=module, reference_mode=reference_mode
    )

    if mode == "off":
        print(f"[GUARD] {side} {symbol}: the historical {opp_side.lower()} reference is off "
              f"for this venue/module (order_guard.conf); price {price} is not compared with past {opp_side.lower()}s")
        return True

    px_val = safe_pos_float(price, default=None)
    prof_pct = safe_float(profit_percentage, default=None)
    if px_val is None or prof_pct is None:
        return False

    if mode == "dynamic":
        trend = (
            getattr(regime_context, "resolved_trend", None)
            if _is_valid_regime_context(regime_context, symbol, provider_name)
            else None
        )
        if trend is None:
            trend = _symbol_trend(symbol, provider=provider)
        dyn_window_s = get_dynamic_window_sec_cfg(
            symbol,
            provider=provider,
            order_type=side,
            resolved_trend=trend,
            regime_context=regime_context,
        )
        dyn_hours = dyn_window_s / 3600.0
        pos_window_ref = safe_pos_float(window_ref, default=None)
        if pos_window_ref is not None:
            diff = profit_diff_pct(side, px_val, pos_window_ref)
            if diff < prof_pct:
                if hasattr(provider, "get_orders"):
                    try:
                        recent = provider.get_orders(symbol, opp_side, dyn_window_s)
                    except (ProviderError, Exception) as exc:
                        print(f"[GUARD] {side} {symbol}: dynamic window history unavailable ({exc}) -> BLOCKED")
                        return False
                    if recent is None:
                        print(f"[GUARD] {side} {symbol}: dynamic window history unavailable -> BLOCKED")
                        return False
                    resolved_dyn_ref = _resolve_window_ref_from_orders(
                        provider, symbol, side, dyn_window_s, recent, eff_mode="strict"
                    )
                    if resolved_dyn_ref is None:
                        print(f"[GUARD] {side} {symbol}: dynamic window ({dyn_hours:.1f}h, trend='{trend}') has no fills; "
                              f"anchor {pos_window_ref} from older period bypassed")
                        return True
                    pos_window_ref = resolved_dyn_ref
                    diff = profit_diff_pct(side, px_val, pos_window_ref)
                elif _is_favorable_trend(side, trend):
                    print(f"[GUARD] {side} {symbol}: dynamic mode bypassed historical {opp_side.lower()} reference "
                          f"during confirmed {trend.upper()} trend; price {px_val} permitted")
                    return True
                if diff < prof_pct:
                    print(
                        f"[GUARD] {side} {symbol}: dynamic mode ({dyn_hours:.1f}h window, trend='{trend}') "
                        f"found recent {opp_side.lower()} ref {pos_window_ref}, price {px_val}, diff {diff:.2f}%, threshold {prof_pct}%\n"
                        f"Percentage difference ({diff:.2f}%) below threshold {prof_pct}%. The {side} order is BLOCKED."
                    )
                    return False
            print(f"[GUARD] {side} {symbol}: dynamic mode ({dyn_hours:.1f}h window, trend='{trend}') "
                  f"recent {opp_side.lower()} ref {pos_window_ref}, price {px_val}, diff {diff:.2f}% >= {prof_pct}%. Permitted.")
            return True
        else:
            print(f"[GUARD] {side} {symbol}: dynamic mode ({dyn_hours:.1f}h window, trend='{trend}') "
                  f"has no recent {opp_side.lower()} reference; price {px_val} permitted")
            return True

    if (
        (reference_mode is not None or module is not None or mode in ("average", "last", "hybrid"))
        and window_ref is None
        and hasattr(provider, "get_orders")
    ):
        try:
            window_ref = window_reference(
                provider, symbol, side, safe_nonneg_float(_venue_cfg(provider_name, "_safeback_sec"), 120.0),
                reference_mode=mode, module=module,
            )
        except Exception:
            pass

    raw_ref = (
        window_ref
        if window_ref is not None
        else (provider.last_opposite_fill(symbol, side) if hasattr(provider, "last_opposite_fill") else None)
    )

    ref = safe_pos_float(raw_ref, default=None)
    if ref is None:
        return True
    diff = profit_diff_pct(side, px_val, ref)
    src = "the window" if window_ref is not None else "the provider"
    print(f"[GUARD] {side} {symbol} (mode={mode}): ref {ref} ({src}), price {px_val}, "
          f"diff {diff:.2f}%, threshold {prof_pct}%")
    if diff < prof_pct:
        print(f"Percentage difference ({diff:.2f}%) below threshold {prof_pct}%. The {side} order is BLOCKED.")
        return False
    return True


def unpack_daily_limit_result(
    limit_result: Any, default_reason: str = "daily_limit"
) -> Tuple[bool, Optional[str]]:
    """Unpack a boolean or ``(allowed, reason)`` tuple from ``daily_limit_guard``."""
    if isinstance(limit_result, (tuple, list)):
        if not limit_result:
            return False, default_reason
        allowed = bool(_parse_bool_flag(limit_result[0], default=False))
        raw_reason: Any = None
        if len(limit_result) >= 2:
            if isinstance(limit_result[1], str) or len(limit_result) < 4:
                raw_reason = limit_result[1]
            else:
                raw_reason = limit_result[3]
        reason_str = str(raw_reason).strip() if raw_reason is not None else ""
        return (True, reason_str or None) if allowed else (False, reason_str or default_reason)
    allowed = bool(_parse_bool_flag(limit_result, default=False))
    return (True, None) if allowed else (False, default_reason)


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
    evaluate_intelligence: bool = True,
    bypass_guards: Optional[set[str]] = None,
    reference_mode: Optional[str] = None,
    module: Optional[str] = None,
):
    """Return whether the order is profitable relative to its reference.

    Execution Pipeline:
    - Emergency Stop-Loss exits or bypass_guards with 'all' bypass all guards.
    - Stage 1 (Mathematical Reference): Deterministic profit margin vs historical reference.
      Bypassed if bypass_guards contains 'math_reference' (or 'math_profit').
    - Stage 2 (Intelligence, Macro & Sentiment): Evaluates Parabolic, Weibull, Gemini LLM,
      and Geopolitical shock shield only on mathematically cleared orders when
      evaluate_intelligence=True.
    """
    bypassed = normalize_bypassed_guards(bypass_guards=bypass_guards)
    if is_master_bypass(bypassed):
        return True

    # Stage 1: Deterministic Mathematical Reference Check (G_PROF)
    if not GUARD_REGISTRY["G_PROF"].is_bypassed(bypassed):
        cm_kwargs = dict(window_ref=window_ref, regime_context=regime_context)
        if reference_mode is not None:
            cm_kwargs["reference_mode"] = reference_mode
        if module is not None:
            cm_kwargs["module"] = module
        if not check_math_profit_reference(
            provider, symbol, order_type, price, profit_percentage,
            **cm_kwargs,
        ):
            return False

    has_valid_qty = safe_pos_float(qty, 0.0) > 0.0
    has_valid_notional = safe_pos_float(notional_eur, 0.0) > 0.0
    if not _parse_bool_flag(evaluate_intelligence, True) or (not has_valid_qty and not has_valid_notional):
        return True

    # Stage 2: Intelligence, Macro & Sentiment Guards
    intel_ok, intel_reason, suggested_scale = check_intelligence_guards(
        provider, symbol, order_type, price, regime_context=regime_context, qty=qty, notional_eur=notional_eur,
        bypass_guards=bypassed,
    )
    if not intel_ok:
        print(f"[GUARD] {order_type} {symbol}: blocked by intelligence guard ({intel_reason})")
        return False

    s_scale = safe_pos_float(suggested_scale, default=1.0)
    if regime_context is not None and s_scale < 1.0:
        _record_context_suggested_scale(
            regime_context,
            normalize_side(order_type),
            s_scale,
            intel_reason,
            reason_attr="intelligence_scale_reason",
            only_if_smaller=True,
        )

    return True


from intelligence.internal.guards.guard_decision import (
    OrderGuardEvaluatorBase,
    OrderGuardEvaluatorFactory,
)


class MathProfitReferenceGuardEvaluator(OrderGuardEvaluatorBase):
    """Concrete evaluator (Level 3) for Stage-1 deterministic mathematical profit reference (`G_PROF`)."""

    cls_name = "MathProfitReferenceGuardEvaluator"
    GUARD_ID = "math_profit_reference"

    def evaluate(
        self,
        provider: Any,
        symbol: str,
        order_type: str,
        price: float,
        profit_percentage: float,
        window_ref: Any = None,
        *,
        regime_context: Any = None,
        reference_mode: Optional[str] = None,
        module: Optional[str] = None,
    ) -> bool:
        return check_math_profit_reference(
            provider,
            symbol,
            order_type,
            price,
            profit_percentage,
            window_ref=window_ref,
            regime_context=regime_context,
            reference_mode=reference_mode,
            module=module,
        )


class IntelligenceMacroSentimentGuardEvaluator(OrderGuardEvaluatorBase):
    """Concrete evaluator (Level 3) for Stage-2 Intelligence, Macro & Sentiment guards."""

    cls_name = "IntelligenceMacroSentimentGuardEvaluator"
    GUARD_ID = "intelligence_guards"

    def evaluate(
        self,
        provider: Any,
        symbol: str,
        order_type: str,
        price: float,
        *,
        regime_context: Any = None,
        qty: Optional[float] = None,
        notional_eur: Optional[float] = None,
        bypass_guards: Optional[set[str]] = None,
    ) -> Tuple[bool, str, float]:
        return check_intelligence_guards(
            provider,
            symbol,
            order_type,
            price,
            regime_context=regime_context,
            qty=qty,
            notional_eur=notional_eur,
            bypass_guards=bypass_guards,
        )


class CompositeProfitGuardEvaluator(OrderGuardEvaluatorBase):
    """Concrete composite evaluator (Level 3) orchestrating Stage-1 math reference and Stage-2 intelligence guards."""

    cls_name = "CompositeProfitGuardEvaluator"
    GUARD_ID = "profit_guard"

    def evaluate(
        self,
        provider: Any,
        symbol: str,
        order_type: str,
        price: float,
        profit_percentage: float,
        window_ref: Any = None,
        *,
        regime_context: Any = None,
        qty: Optional[float] = None,
        notional_eur: Optional[float] = None,
        evaluate_intelligence: bool = True,
        bypass_guards: Optional[set[str]] = None,
        reference_mode: Optional[str] = None,
        module: Optional[str] = None,
    ) -> bool:
        return profit_guard(
            provider,
            symbol,
            order_type,
            price,
            profit_percentage,
            window_ref=window_ref,
            regime_context=regime_context,
            qty=qty,
            notional_eur=notional_eur,
            evaluate_intelligence=evaluate_intelligence,
            bypass_guards=bypass_guards,
            reference_mode=reference_mode,
            module=module,
        )


for _og_cls in (
    MathProfitReferenceGuardEvaluator,
    IntelligenceMacroSentimentGuardEvaluator,
    CompositeProfitGuardEvaluator,
):
    OrderGuardEvaluatorFactory.register(_og_cls.GUARD_ID, _og_cls)

from providers.base import MarketDataProvider as _MarketDataProvider  # noqa: E402

_MarketDataProvider.bind_order_guard_hooks(
    weight_limit_fn=lambda *a, **kw: weight_limit(*a, **kw),
    window_cfg_fn=lambda *a, **kw: get_window_cfg(*a, **kw),
    window_reference_fn=lambda *a, **kw: window_reference(*a, **kw),
    resolve_trade_weight_fn=lambda *a, **kw: resolve_trade_weight(*a, **kw),
    compute_weight_capped_qty_fn=lambda *a, **kw: compute_weight_capped_qty(*a, **kw),
)

